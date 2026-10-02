import os
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import argparse
import json
import math

import cv2
import numpy as np
from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, utils
from tqdm import tqdm
from waymo.box_utils import bbox_to_corner3d, get_bound_2d_mask

"""Saved extrinsics already include the OpenCV axis conversion."""
camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: "FRONT_LEFT",
    dataset_pb2.CameraName.FRONT_RIGHT: "FRONT_RIGHT",
    dataset_pb2.CameraName.FRONT: "FRONT",
    dataset_pb2.CameraName.SIDE_LEFT: "SIDE_LEFT",
    dataset_pb2.CameraName.SIDE_RIGHT: "SIDE_RIGHT",
}

laser_names_dict = {
    dataset_pb2.LaserName.TOP: "TOP",
    dataset_pb2.LaserName.FRONT: "FRONT",
    dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
    dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
    dataset_pb2.LaserName.REAR: "REAR",
}

WAYMO_CLASSES = ["unknown", "Vehicle", "Pedestrian", "Sign", "Cyclist"]
WAYMO_DYNAMIC_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]
opencv2camera = np.array([[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def project_numpy(xyz, K, RT, H, W):
    """
    input:
    xyz: [N, 3], pointcloud
    K: [3, 3], intrinsic
    RT: [4, 4], w2c

    output:
    mask: [N], pointcloud in camera frustum
    xy: [N, 2], coord in image plane
    """

    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)

    return xyz_pixel, mask


def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)  # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera)  # [forward, left, up] to [right, down, forward]
    return extrinsic


def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic


def project_label_to_image(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], K=intrinsic, RT=np.linalg.inv(extrinsic), H=height, W=width
    )
    return points_uv, valid


def project_label_to_mask(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T  # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3], K=intrinsic, pose=np.linalg.inv(extrinsic), H=height, W=width
    )

    return mask


def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, start_idx=None, end_idx=None):
    print(f"Processing sequence {seq_name}...")
    print(f"Saving to {seq_save_dir}")

    castrack_infos = json.loads(Path(track_file).read_text()) if track_file else {}
    if track_file and seq_name not in castrack_infos:
        raise KeyError(f"No predictions for {seq_name} in {track_file}")

    os.makedirs(seq_save_dir, exist_ok=True)

    seq_path = os.path.join(root_dir, seq_name + ".tfrecord")

    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    start_idx = start_idx or 0
    end_idx = num_frames - 1 if end_idx is None else min(end_idx, num_frames - 1)

    if "pose" in process_list:
        ego_pose_save_dir = os.path.join(seq_save_dir, "ego_pose")
        os.makedirs(ego_pose_save_dir, exist_ok=True)
        print("Processing ego pose...")
        timestamp = dict()
        timestamp["FRAME"] = dict()
        for camera_name in camera_names_dict.values():
            timestamp[camera_name] = dict()

        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
            pose = np.array(frame.pose.transform).reshape(4, 4)
            np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}.txt"), pose)
            timestamp["FRAME"][str(frame_id).zfill(6)] = frame.timestamp_micros / 1e6

            camera_calibrations = frame.context.camera_calibrations
            for i, camera in enumerate(camera_calibrations):
                camera_name = camera.name
                camera_name_str = camera_names_dict[camera_name]
                camera = utils.get(frame.images, camera_name)
                camera_timestamp = camera.pose_timestamp
                timestamp[camera_name_str][str(frame_id).zfill(6)] = camera_timestamp

                camera_pose = np.array(camera.pose.transform).reshape(4, 4)
                np.savetxt(
                    os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}_{camera_name - 1}.txt"), camera_pose
                )

        timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
        with open(timestamp_save_path, "w") as f:
            json.dump(timestamp, f, indent=1)

    if "calib" in process_list:
        intrinsic_save_dir = os.path.join(seq_save_dir, "intrinsics")
        extrinsic_save_dir = os.path.join(seq_save_dir, "extrinsics")
        os.makedirs(intrinsic_save_dir, exist_ok=True)
        os.makedirs(extrinsic_save_dir, exist_ok=True)
        print("Processing camera calibration...")

        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
            camera_calibrations = frame.context.camera_calibrations

        extrinsics = []
        intrinsics = []
        camera_names = []
        for camera in camera_calibrations:
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            extrinsic = np.matmul(extrinsic, opencv2camera)  # [forward, left, up] to [right, down, forward]
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            camera_names.append(camera.name)

        for i in range(5):
            np.savetxt(os.path.join(extrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), extrinsics[i])
            np.savetxt(os.path.join(intrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), intrinsics[i])

    image_save_dir = os.path.join(seq_save_dir, "images")
    if "image" in process_list and not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir, exist_ok=True)
        print("Processing image data...")

        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
            for camera_name, camera_name_str in camera_names_dict.items():
                camera = utils.get(frame.images, camera_name)
                img = utils.decode_image(camera)
                img_path = os.path.join(image_save_dir, f"{frame_id:06d}_{str(camera.name - 1)}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print("Processing image data done...")

    if "lidar" in process_list:
        pts_3d_all = dict()
        pts_2d_all = dict()
        print("Processing LiDAR data...")
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
            lidar_rows = []
            pts_3d = []  # LiDAR point cloud in world frame
            pts_2d = []  # LiDAR point cloud projection in camera [camera_name, w, h]

            for laser_name, laser_name_str in laser_names_dict.items():
                laser = utils.get(frame.lasers, laser_name)
                laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
                ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

                pcl, pcl_attr = utils.project_to_pointcloud(
                    frame, ri, camera_projection, range_image_pose, laser_calibration
                )

                rows = np.zeros((len(pcl), 14), dtype=np.float32)
                rows[:, :3] = np.array(laser_calibration.extrinsic.transform).reshape(4, 4)[:3, 3]
                rows[:, 3:6] = pcl[:, :3]
                rows[:, 11:13] = pcl_attr[:, 1:3]
                rows[:, 13] = laser_name - 1
                lidar_rows.append(rows)
                pts_3d.append(pcl[:, :3])  # save LiDAR pointcloud in vehicle frame

                mask = ri[:, :, 0] > 0
                camera_projection = camera_projection[mask]

                camera_projection[:, 0] -= 1
                camera_projection[:, 3] -= 1
                camera_projection = camera_projection.astype(np.int16)

                pts_2d.append(camera_projection)

            pts_3d = np.concatenate(pts_3d, axis=0)
            lidar_dir = Path(seq_save_dir) / "lidar"
            lidar_dir.mkdir(exist_ok=True)
            np.concatenate(lidar_rows).tofile(lidar_dir / f"{frame_id:03d}.bin")
            pts_3d_all[frame_id] = pts_3d
            pts_2d = np.concatenate(pts_2d, axis=0)
            pts_2d_all[frame_id] = pts_2d

        np.savez_compressed(f"{seq_save_dir}/pointcloud.npz", pointcloud=pts_3d_all, camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")

    if "track" in process_list:
        print("Processing tracking data...")
        track_dir = os.path.join(seq_save_dir, "track")
        os.makedirs(track_dir, exist_ok=True)

        if seq_name in castrack_infos:
            track_infos_path = os.path.join(track_dir, "track_info_castrack.txt")
            track_infos_file = open(track_infos_path, "w")
            row_info_title = (
                "frame_id "
                + "track_id "
                + "object_class "
                + "alpha "
                + "box_height "
                + "box_width "
                + "box_length "
                + "box_center_x "
                + "box_center_y "
                + "box_center_z "
                + "box_heading "
                + "\n"
            )
            track_infos_file.write(row_info_title)

            track_info = castrack_infos[seq_name]
            bbox_visible_dict = dict()
            object_ids = dict()

            datafile = WaymoDataFileReader(seq_path)
            for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
                label = track_info[str(frame_id)]
                for i, object_id in enumerate(label["obj_ids"]):
                    box = label["boxes_lidar"][i]

                    length, width, height = box[3], box[4], box[5]

                    tx, ty, tz = box[0], box[1], box[2]
                    heading = box[-1]
                    c = math.cos(heading)
                    s = math.sin(heading)
                    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = rotz_matrix
                    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                    if object_id not in object_ids:
                        object_ids[object_id] = len(object_ids)

                    label_id = object_ids[object_id]
                    if label_id not in bbox_visible_dict:
                        bbox_visible_dict[label_id] = dict()
                    bbox_visible_dict[label_id][frame_id] = []

                    for camera_name in camera_names_dict.keys():
                        if camera_name == dataset_pb2.CameraName.FRONT:
                            camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)

                            vertices, valid = project_label_to_image(
                                dim=[length, width, height],
                                obj_pose=obj_pose_vehicle,
                                calibration=camera_calibration,
                            )

                            if valid.any():
                                bbox_visible_dict[label_id][frame_id].append(camera_name - 1)
                    bbox_visible_dict[label_id][frame_id] = sorted(bbox_visible_dict[label_id][frame_id])

                    name = label["name"][i]
                    if name == "Vehicle":
                        obj_class = "vehicle"
                    elif name == "Cyclist":
                        obj_class = "cyclist"
                    elif name == "Pedestrian":
                        obj_class = "pedestrian"
                    else:
                        obj_class = "misc"

                    alpha = -10

                    lines_info = f"{frame_id} {label_id} {obj_class} {alpha} {height} {width} {length} {tx} {ty} {tz} {heading} \n"

                    track_infos_file.write(lines_info)

            bbox_visible_path = os.path.join(track_dir, "track_camera_vis_castrack.json")
            with open(bbox_visible_path, "w") as f:
                json.dump(bbox_visible_dict, f, indent=1)

            object_ids_path = os.path.join(track_dir, "track_ids_castrack.json")
            with open(object_ids_path, "w") as f:
                json.dump(object_ids, f, indent=2)

            track_infos_file.close()
            print("Processing tracking data done...")

    if "dynamic_mask" in process_list:
        print("Saving dynamic mask ...")
        dynamic_mask_dir = os.path.join(seq_save_dir, "dynamic_mask")
        os.makedirs(dynamic_mask_dir, exist_ok=True)
        datafile = WaymoDataFileReader(seq_path)

        for frame_id, frame in tqdm(enumerate(islice(datafile, end_idx + 1))):
            masks = dict()
            for camera_name in camera_names_dict.keys():
                camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                width, height = camera_calibration.width, camera_calibration.height
                mask = np.zeros((height, width), dtype=np.uint8)
                masks[camera_name] = mask

            for label in frame.laser_labels:
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])

                if speed < 1.0:
                    continue

                length, width, height = box.length, box.width, box.height

                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c = math.cos(heading)
                s = math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                for camera_name in camera_names_dict.keys():
                    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                    dim = [length, width, height]
                    vertices, valid = project_label_to_image(
                        dim=dim,
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    if valid.any():
                        mask = project_label_to_mask(
                            dim=dim,
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                        masks[camera_name] = np.logical_or(masks[camera_name], mask)

            for camera_name in camera_names_dict.keys():
                mask = masks[camera_name]
                mask_path = os.path.join(dynamic_mask_dir, f"{frame_id:06d}_{str(camera_name - 1)}.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        print("Saving dynamic mask done...")

    if "objects" in process_list:
        """Parse and save the ground truth bounding boxes."""
        print("Processing objects data...")
        instances_info, frame_instances = {}, {}
        dataset = WaymoDataFileReader(seq_path)
        for frame_idx, frame in enumerate(islice(dataset, end_idx + 1)):
            frame_instances[frame_idx] = []
            for l in frame.laser_labels:
                frame_pose = np.array(frame.pose.transform).reshape(4, 4)

                str_id = str(l.id)
                if WAYMO_CLASSES[l.type] not in WAYMO_DYNAMIC_CLASSES:
                    continue

                frame_instances[frame_idx].append(str_id)

                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=l.id,
                        class_name=WAYMO_CLASSES[l.type],
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        },
                    )

                box = l.box

                tx, ty, tz = box.center_x, box.center_y, box.center_z

                c = math.cos(box.heading)
                s = math.sin(box.heading)

                o2v = np.array([[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

                pose = frame_pose @ o2v  # o2w = v2w @ o2v

                dimension = [box.length, box.width, box.height]

                instances_info[str_id]["frame_annotations"]["frame_idx"].append(frame_idx)
                instances_info[str_id]["frame_annotations"]["obj_to_world"].append(pose.tolist())
                instances_info[str_id]["frame_annotations"]["box_size"].append(dimension)

        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]

        obj_save_dir = os.path.join(seq_save_dir, "objects")
        os.makedirs(obj_save_dir, exist_ok=True)

        with open(f"{obj_save_dir}/instances_info.json", "w") as fp:
            json.dump(new_instances_info, fp, indent=4)
        with open(f"{obj_save_dir}/frame_instances.json", "w") as fp:
            json.dump(new_frame_instances, fp, indent=4)

    print("Processing objects data done...")


def main():
    parser = argparse.ArgumentParser(description="Extract Waymo TFRecords for EvolSplat4D without TensorFlow.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--scene-ids", type=int, nargs="+", required=True)
    parser.add_argument("--track-file", type=Path, help="Optional CasTrack prediction JSON for dynamic inference")
    parser.add_argument("--max-frames", type=int, help="Extract only the first N frames for a smoke check")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")
    raw_root, output_root = args.raw_root.resolve(), args.output_root.resolve()
    if output_root == raw_root or output_root.is_relative_to(raw_root):
        parser.error("output-root must be outside the read-only raw dataset")
    segments = (Path(__file__).parent / "splits" / f"segment_list_{args.split}.txt").read_text().splitlines()
    for scene_id in args.scene_ids:
        if not 0 <= scene_id < len(segments):
            parser.error(f"Invalid {args.split} scene ID: {scene_id}")
        source = raw_root / (segments[scene_id] + ".tfrecord")
        if not source.is_file():
            parser.error(f"Missing TFRecord: {source}")
        target = output_root / f"{scene_id:03d}"
        target.mkdir(parents=True, exist_ok=False)
        keys = ["pose", "calib", "image", "lidar", "dynamic_mask", "objects"]
        if args.track_file:
            keys.append("track")
        parse_seq_rawdata(
            keys,
            str(raw_root),
            segments[scene_id],
            str(target),
            args.track_file,
            end_idx=args.max_frames - 1 if args.max_frames else None,
        )


if __name__ == "__main__":
    main()
