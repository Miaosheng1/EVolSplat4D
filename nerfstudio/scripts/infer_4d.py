# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Render novel Waymo scenes with EvolSplat4D and report PSNR, SSIM and LPIPS."""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import mediapy as media
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import AnnotatedBaseConfigUnion
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.utils import colormaps, comms, profiler
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    """Finds a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    """Feedforward Inference function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """

    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup_feedforward()
    trainer.pipeline.eval()

    ## Init the volume and offset for each GS point
    trainer.pipeline.model.init_volume()
    num_eval_images = len(trainer.pipeline.datamanager.eval_dataset)
    assert config.experiment_name is not None
    save_dir = "Zeroshot/extrap/" + config.experiment_name
    if Path(save_dir).is_dir():
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    lpips_fn = LearnedPerceptualImagePatchSimilarity().cuda()
    ssim = structural_similarity_index_measure

    sum_psnr = 0.0
    sum_lpips = 0.0
    sum_ssim = 0.0
    per_image_results = []

    for i in tqdm(range(num_eval_images)):
        with torch.no_grad():
            outputs, gt_img = trainer.pipeline.get_eval_zeroshot(camera_idx=i)

        render_img = outputs["rgb"]
        rigid_img = outputs["rigid_rgb"]
        depth_map = colormaps.apply_XDLab_color_depth_map(outputs["depth"].squeeze().detach().cpu().numpy())
        gt_img = torch.tensor(gt_img).to(outputs["accumulation"])

        media.write_image(os.path.join(save_dir, "render_{:03d}.png".format(i)), render_img.detach().cpu().numpy())
        media.write_image(os.path.join(save_dir, "rigid_{:03d}.png".format(i)), rigid_img.detach().cpu().numpy())
        media.write_image(
            os.path.join(save_dir, "background_{:03d}.png".format(i)), outputs["background"].detach().cpu().numpy()
        )
        media.write_image(os.path.join(save_dir, "depth_{:03d}.png".format(i)), depth_map)

        psnr = -10.0 * np.log10(np.mean(np.square(gt_img.detach().cpu().numpy() - render_img.detach().cpu().numpy())))
        lpips_val = float(lpips_fn(gt_img.permute(0, 3, 1, 2), render_img.unsqueeze(0).permute(0, 3, 1, 2)).item())
        ssim_val = float(ssim(render_img.unsqueeze(0).permute(0, 3, 1, 2), gt_img.permute(0, 3, 1, 2)).item())
        print("image {} PSNR:{:.4f} ,SSIM: {:.4f}, LPIPS: {:.4f}".format(i, psnr, ssim_val, lpips_val))

        sum_psnr += float(psnr)
        sum_lpips += lpips_val
        sum_ssim += ssim_val

        per_image_results.append(
            {
                "index": i,
                "image": f"render_{i:03d}.png",
                "psnr": float(psnr),
                "ssim": ssim_val,
                "lpips": lpips_val,
            }
        )

    avg_psnr = sum_psnr / num_eval_images
    avg_ssim = sum_ssim / num_eval_images
    avg_lpips = sum_lpips / num_eval_images

    CONSOLE.print(f"[bold green]Average PSNR:{avg_psnr}", justify="center")
    CONSOLE.print(f"[bold green]Average SSIM:{avg_ssim}", justify="center")
    CONSOLE.print(f"[bold green]Average Lpips:{avg_lpips}", justify="center")
    CONSOLE.print(f"[bold yellow]Save Dir in :{save_dir}")
    # Write per-image metrics and averages to JSON
    dataset_name = str(config.pipeline.datamanager.dataparser.data.name)
    metrics_output = {
        "data_seq": dataset_name,
        "results": per_image_results,
        "average": {
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "lpips": float(avg_lpips),
        },
    }
    json_path = os.path.join(save_dir, f"{dataset_name}_metric.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(metrics_output, jf, ensure_ascii=False, indent=2)
    CONSOLE.print(f"[bold green]Metrics JSON saved to: {json_path}")


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    """Spawned distributed worker that handles the initialization of process group and handles the
       training process on multiple processes.

    Args:
        local_rank: Current rank of process.
        main_func: Function that will be called by the distributed workers.
        world_size: Total number of gpus available.
        num_devices_per_machine: Number of GPUs per machine.
        machine_rank: Rank of this machine.
        dist_url: URL to connect to for distributed jobs, including protocol
            E.g., "tcp://127.0.0.1:8686".
            It can be set to "auto" to automatically select a free port on localhost.
        config: TrainerConfig specifying training regimen.
        timeout: Timeout of the distributed workers.

    Raises:
        e: Exception in initializing the process group

    Returns:
        Any: TODO: determine the return type
    """
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    """Function that spawns multiple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_devices_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (TrainerConfig, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
        device_type: type of device to use for training.
    """
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    elif world_size == 1:
        # uses one process
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())
        finally:
            profiler.flush_profiler(config.logging)
    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            CONSOLE.log("file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://")

        process_context = mp.spawn(
            _distributed_worker,
            nprocs=num_devices_per_machine,
            join=False,
            args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
        )
        # process_context won't be None because join=False, so it's okay to assert this
        # for Pylance reasons
        assert process_context is not None
        try:
            process_context.join()
        except KeyboardInterrupt:
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    CONSOLE.log(f"Terminating process {i}...")
                    process.terminate()
                process.join()
                CONSOLE.log(f"Process {i} finished.")
        finally:
            profiler.flush_profiler(config.logging)


def main(config: TrainerConfig) -> None:
    """Main function."""

    if config.data:
        CONSOLE.log("Using --data alias for --data.pipeline.datamanager.data")
        config.pipeline.datamanager.data = config.data

    if config.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.load_config}")
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()

    config_file = config.config_file
    if os.path.exists(config_file):  # type: ignore
        file = OmegaConf.load(config_file)
        opts = edict(file)
        config.fuse_config(Manner_config=opts)
    else:
        CONSOLE.log(f"No mannual config file found at: {config_file}")

    launch(
        main_func=infer_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )  # type: ignore
    )


if __name__ == "__main__":
    entrypoint()
