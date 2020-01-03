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

import paddle.fluid as fluid
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg


class EAST(object):
    def __init__(self,
                 add_conv_body_func=None,
                 mode='train',
                 use_random=True,
                 text_scale=512):
        self.add_conv_body_func = add_conv_body_func
        self.mode = mode
        self.use_random = use_random
        self.text_scale = text_scale

    def unpool(self, inputs):
        #return fluid.layers.resize_bilinear(inputs, out_shape=[inputs.shape[2]*2, inputs.shape[3]*2])
        return fluid.layers.resize_bilinear(inputs, scale=2)
    def build_model(self, image_shape):
        self.build_input(image_shape)
        logits, end_points = self.add_conv_body_func(self.image)
        f = end_points[::-1]
        print(f)
        for i in range(4):
            print('Shape of f_{} {}'.format(i, f[i].shape))
        g = [None, None, None, None]
        h = [None, None, None, None]
        num_outputs = [None, 128, 64, 32]

        for i in range(4):
            if i == 0:
                h[i] = f[i]
            else:
                w_param_attrs_1 = fluid.ParamAttr(learning_rate=1,
                                regularizer=fluid.regularizer.L2Decay(1e-5),
                                trainable=True)
                c1_1 = fluid.layers.conv2d(input=fluid.layers.concat([g[i-1], f[i]], axis=1), num_filters=num_outputs[i], filter_size=1, param_attr=w_param_attrs_1)
                c1_1 = fluid.layers.batch_norm(c1_1, momentum=0.997, epsilon=1e-05, is_test=False)
                c1_1 = fluid.layers.relu(c1_1)
                w_param_attrs_2 = fluid.ParamAttr(learning_rate=1,
                                regularizer=fluid.regularizer.L2Decay(1e-5),
                                trainable=True)
                h[i] = fluid.layers.conv2d(input=c1_1, num_filters=num_outputs[i], filter_size=3, padding=1, param_attr=w_param_attrs_2)
                h[i] = fluid.layers.batch_norm(h[i], momentum=0.997, epsilon=1e-05, is_test=False)
                h[i] = fluid.layers.relu(h[i])
            if i <= 2:
                g[i] = self.unpool(h[i])
            else:
                w_param_attrs_3 = fluid.ParamAttr(learning_rate=1,
                                regularizer=fluid.regularizer.L2Decay(1e-5),
                                trainable=True)
                g[i] = fluid.layers.conv2d(input=h[i], num_filters=num_outputs[i], filter_size=3, padding=1, param_attr=w_param_attrs_3)
                g[i] = fluid.layers.batch_norm(g[i], momentum=0.997, epsilon=1e-05, is_test=False)
                g[i] = fluid.layers.relu(g[i])
            print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
        self.F_score = fluid.layers.conv2d(input=g[3], num_filters=1, filter_size=1, act='sigmoid')
        geo_map = fluid.layers.conv2d(input=g[3], num_filters=4, filter_size=1, act='sigmoid') * self.text_scale
        angle_map = (fluid.layers.conv2d(input=g[3], num_filters=1, filter_size=1, act='sigmoid') - 0.5) * np.pi/2 # angle is between [-45, 45]
        self.F_geometry = fluid.layers.concat([geo_map, angle_map], axis=1)
        return self.F_score, self.F_geometry


    def loss(self):
        losses = []
        # Fast RCNN loss
        model_loss, g_loss, cls_loss = self.get_loss(self.input_score_maps, self.F_score, self.input_geo_maps, self.F_geometry, self.input_training_masks)
        # RPN loss
        rkeys = ['model_loss', 'g_loss', 'cls_loss']
        rloss = [model_loss, g_loss, cls_loss]
        return rloss, rkeys

    def feeds(self):
        if self.mode == 'infer':
            return [self.image, self.im_info]
        if self.mode == 'val':
            return [self.image, self.input_score_maps, self.input_geo_maps, self.input_training_masks]
        return [self.image, self.input_score_maps, self.input_geo_maps, self.input_training_masks]


    def build_input(self, image_shape):
        self.image = fluid.layers.data(
            name='input_images', shape=image_shape, dtype='float32')
        self.input_score_maps = fluid.layers.data(
            name='input_score_maps', shape=[None, None, None, 1], dtype='float32', append_batch_size=False)
        self.input_geo_maps = fluid.layers.data(
            name='input_geo_maps', shape=[None, None, None, 5], dtype='float32', append_batch_size=False)
        self.input_training_masks = fluid.layers.data(
            name='input_training_masks', shape=[None, None, None, 1], dtype='float32', append_batch_size=False)
        self.im_info = fluid.layers.data(
            name='im_info', shape=[None, 4], dtype='float32', append_batch_size=False)
  
    def dice_coefficient(self, y_true_cls, y_pred_cls, training_mask):
        '''
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        '''
        eps = 1e-5
        intersection = fluid.layers.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = fluid.layers.reduce_sum(y_true_cls * training_mask) + fluid.layers.reduce_sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def get_loss(self, y_true_cls, y_pred_cls,
             y_true_geo, y_pred_geo,
             training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = self.dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01
    
        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = fluid.layers.split(input=y_true_geo, num_or_sections=5, dim=1)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = fluid.layers.split(input=y_pred_geo, num_or_sections=5, dim=1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = fluid.layers.elementwise_min(d2_gt, d2_pred, axis=1) + fluid.layers.elementwise_min(d4_gt, d4_pred, axis=1)
        h_union = fluid.layers.elementwise_min(d1_gt, d1_pred, axis=1) + fluid.layers.elementwise_min(d3_gt, d3_pred, axis=1)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = (-1)*fluid.layers.log(fluid.layers.elementwise_div(area_intersect + 1.0, area_union + 1.0))
        L_theta = 1 - fluid.layers.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta
        g_loss = fluid.layers.reduce_mean(L_g * y_true_cls * training_mask)
        total_loss = g_loss + classification_loss
        return total_loss, g_loss, classification_loss
