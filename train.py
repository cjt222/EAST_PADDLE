#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#import icdar
import reader as icdar
def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


set_paddle_flags({
    'FLAGS_conv_workspace_size_limit': 500,
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import sys
import numpy as np
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time, check_gpu
import collections

import paddle
import paddle.fluid as fluid
from paddle.fluid import profiler
#import reader
import models.model_builder as model_builder
import models.resnet as resnet
from learning_rate import exponential_with_warmup_decay
from config import cfg
import checkpoint as checkpoint
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1:
        return 1
    return fluid.core.get_cuda_device_count()


def save_model(exe, postfix, program):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path, program)

def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, 512, 512]

    if cfg.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices_num = get_device_num()
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    use_random = True
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            model = model_builder.EAST(
                add_conv_body_func=resnet.ResNet(),
                use_random=use_random)
            model.build_model(image_shape)
            losses, keys = model.loss()
            loss = losses[0]
            fetch_list = losses

            boundaries = cfg.lr_steps
            gamma = cfg.lr_gamma
            step_num = len(cfg.lr_steps)
            values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

            lr = exponential_with_warmup_decay(
                learning_rate=learning_rate,
                boundaries=boundaries,
                values=values,
                warmup_iter=cfg.warm_up_iter,
                warmup_factor=cfg.warm_up_factor)
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, regularization=fluid.regularizer.L2Decay(cfg.weight_decay))
            optimizer.minimize(loss)
            fetch_list = fetch_list + [lr]

            for var in fetch_list:
                var.persistable = True

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_all_optimizer_ops = False
    build_strategy.fuse_elewise_add_act_ops = True
    build_strategy.sync_batch_norm=True
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1
    exe.run(startup_prog)

    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)


    dataset = icdar.ICDAR2015Dataset()
    data_generator = dataset.get_batch(num_workers=24,
                                     input_size=512,
                                     batch_size=14)

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        train_stats = TrainingStats(cfg.log_window, keys)
        #for iter_id, data in enumerate(next(data_generator)):
        for iter_id in range(100000):
            data = next(data_generator)
            #for data in data_list:
            prev_start_time = start_time
            start_time = time.time()
            outs = exe.run(compiled_train_prog, fetch_list=[v.name for v in fetch_list],
                                 feed={"input_images": data[0],
                                       "input_score_maps": data[2],
                                       "input_geo_maps": data[3],
                                       "input_training_masks": data[4]})
            stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
            train_stats.update(stats)
            logs = train_stats.log()
            strs = '{}, batch: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                now_time(), iter_id,
                np.mean(outs[-1]), logs, start_time - prev_start_time)
            if iter_id % 10 == 0:
                print(strs)
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model(exe, "model_iter{}".format(iter_id), train_prog)
            if (iter_id + 1) == cfg.max_iter:
                break
        end_time = time.time()
        total_time = end_time - start_time
        last_loss = np.array(outs[0]).mean()
    train_loop()


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    train()
