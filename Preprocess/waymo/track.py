import json
import math
import os

import numpy as np
import torch
from waymo.waymo_utils import matrix_to_quaternion

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}


def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world


def get_obj_pose_tracking(datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []
    objects_info = {}

    tracklet_path = os.path.join(datadir, "track/track_info_castrack.txt")
    tracklet_camera_vis_path = os.path.join(datadir, "track/track_camera_vis_castrack.json")

    print(f"Loading from : {tracklet_path}")
    f = open(tracklet_path, "r")
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]

    f = open(tracklet_camera_vis_path, "r")
    tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, "images")
    n_cameras = 5
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)

    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]

        if object_class in ["sign", "misc"]:
            continue

        cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
        join_cameras_list = list(set(cameras) & set(cameras_vis_list))
        if len(join_cameras_list) == 0:
            continue

        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]["track_id"] = track_id
            objects_info[track_id]["class"] = object_class
            objects_info[track_id]["class_label"] = waymo_track2label[object_class]
            objects_info[track_id]["height"] = float(tracklet[4])
            objects_info[track_id]["width"] = float(tracklet[5])
            objects_info[track_id]["length"] = float(tracklet[6])
        else:
            objects_info[track_id]["height"] = max(objects_info[track_id]["height"], float(tracklet[4]))
            objects_info[track_id]["width"] = max(objects_info[track_id]["width"], float(tracklet[5]))
            objects_info[track_id]["length"] = max(objects_info[track_id]["length"], float(tracklet[6]))

        tr_array = np.concatenate(
            [
                np.array(tracklet[:2]).astype(np.float64),
                np.array([waymo_track2label[object_class]]),
                np.array(tracklet[4:]).astype(np.float64),
            ]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1

    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame : end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # Remove static objects
    print("Removing static objects")
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_obj_idx] = -1.0
                visible_objects_pose_vehicle[all_obj_idx] = -1.0
                visible_objects_pose_world[all_obj_idx] = -1.0
                objects_info.pop(key)
        else:
            objects_info.pop(key)

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    # print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving obj in current sequence, make dummy visible objects")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = 1.0
    print("box scale: ", box_scale)

    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj["class"] == "pedestrian":
            obj["deformable"] = True
        else:
            obj["deformable"] = False

        obj["width"] = obj["width"] * box_scale
        obj["length"] = obj["length"] * box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj["start_frame"] = np.min(obj_frames)
        obj["end_frame"] = np.max(obj_frames)

        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate([visible_objects_ids[..., None], visible_objects_pose_world], axis=-1)

    objects_tracklets_vehicle = np.concatenate([visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1)

    return objects_tracklets_world, objects_tracklets_vehicle, objects_info


def inverse_rigid_trans(Tr):
    """Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr
