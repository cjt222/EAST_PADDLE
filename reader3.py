# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import random
import numpy as np
import xml.etree.ElementTree
import os
import time
import copy
import six
import cv2
import math
import paddle
from collections import deque

import data_utils
from roidbs import ICDAR2015Dataset, ICDAR2017Dataset
from config import cfg
from PIL import Image
from data_utils import _resize
num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
np.random.seed(10)


def roidb_reader(roidb, mode):
    im, im_scales, gt_boxes, gt_classes = data_utils.get_image_blob(roidb, mode)
    im_id = roidb['im_id']
    is_crowd = roidb['is_crowd']
    im_height = np.round(roidb['height'] * im_scales)
    im_width = np.round(roidb['width'] * im_scales)
    is_difficult = roidb['is_difficult']
    im_info = np.array([im_height, im_width, im_scales], dtype=np.float32)
    if mode == 'val':
        return im, gt_boxes, gt_classes, is_crowd, im_info, im_id, is_difficult

    outs = (im, gt_boxes, gt_classes, is_crowd, im_info, im_id)

    return outs


def EASTData(mode,
             batch_size=None,
             total_batch_size=None,
             padding_total=False,
             shuffle=False,
             shuffle_seed=None):  #,
    #roidbs=None):
    total_batch_size = total_batch_size if total_batch_size else batch_size
    assert total_batch_size % batch_size == 0
    icdar2015_dataset = ICDAR2015Dataset(mode)
    roidbs = icdar2015_dataset.get_roidb()

    print("{} on {} with {} roidbs".format(mode, cfg.dataset, len(roidbs)))

    def reader():
        if mode == "train":
            if shuffle:
                if shuffle_seed is not None:
                    np.random.seed(shuffle_seed)
                roidb_perm = deque(np.random.permutation(roidbs))
            else:
                roidb_perm = deque(roidbs)
            roidb_cur = 0
            count = 0
            batch_out = []
            device_num = total_batch_size / batch_size
            while True:
                start = time.time()
                roidb = roidb_perm[0]
                roidb_cur += 1
                roidb_perm.rotate(-1)
                if roidb_cur >= len(roidbs):
                    if shuffle:
                        roidb_perm = deque(np.random.permutation(roidbs))
                    else:
                        roidb_perm = deque(roidbs)
                    roidb_cur = 0
                # im, gt_boxes, gt_classes, is_crowd, im_info, im_id, gt_masks
                datas = roidb_reader(roidb, mode)
                if datas[1].shape[0] == 0:
                    continue
                batch_out.append(datas)
                end = time.time()
                #print('reader time:', end - start)
                if len(batch_out) == batch_size:
                    yield batch_out
                    count += 1
                    batch_out = []
                iter_id = count // device_num
                if iter_id >= cfg.max_iter * num_trainers:
                    return
        elif mode == "val":
            batch_out = []
            for roidb in roidbs:
                im, gt_boxes, gt_classes, is_crowd, im_info, im_id, is_difficult = roidb_reader(
                    roidb, mode)
                batch_out.append((im, gt_boxes, gt_classes, is_crowd, im_info,
                                  im_id, is_difficult))
                if len(batch_out) == batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) != 0:
                yield batch_out

    return reader


def train(batch_size,
          total_batch_size=None,
          padding_total=False,
          num_workers=20,
          shuffle=True,
          shuffle_seed=None):
    return RRPNData(
        'train',
        batch_size,
        total_batch_size,
        padding_total,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed)


def test(batch_size, total_batch_size=None, padding_total=False):
    return RRPNData('val', batch_size, total_batch_size, shuffle=False)
