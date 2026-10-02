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

"""EvolSplat4D training and inference configuration."""

import tyro

from nerfstudio.data.datamanagers.evolsplat4d_datamanager import EvolSplat4DDataManagerConfig
from nerfstudio.data.dataparsers.evolsplat4d_dataparser import EvolSplat4DDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.evolsplat4d import EvolSplat4DModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

method_configs = {}
method_configs["evolsplat4d"] = TrainerConfig(
    method_name="evolsplat4d",
    steps_per_eval_image=500,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=50000000,
    max_num_iterations=10000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=EvolSplat4DDataManagerConfig(
            dataparser=EvolSplat4DDataParserConfig(load_3D_points=True)  ## pretrain on multi-scene
        ),
        model=EvolSplat4DModelConfig(),
    ),
    optimizers={
        "sparse_conv": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=500, lr_pre_warmup=0
            ),
        },
        "mlp_conv": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "mlp_opacity": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "mlp_offset": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "gaussianDecoder": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "background_model": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.00001, max_steps=30000),
        },
        "rigid_decoder": {
            "optimizer": AdamOptimizerConfig(lr=1 * 1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
    },
    vis="tensorboard",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.conf.FlagConversionOff[tyro.extras.subcommand_type_from_defaults(defaults=method_configs)]
]
