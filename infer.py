#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import time
import numpy as np
import cv2
import paddle
import paddle.fluid as fluid
import reader2 as reader
from utility import print_arguments, parse_args, check_gpu
import models.model_builder as model_builder
import models.resnet as resnet
import locality_aware_nms as nms_locality
from config import cfg
from icdar import restore_rectangle
import lanms
from eval_helper import *


#def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
#    '''
#    restore text boxes from score map and geo map
#    :param score_map:
#    :param geo_map:
#    :param timer:
#    :param score_map_thresh: threshhold for score map
#    :param box_thresh: threshhold for boxes
#    :param nms_thres: threshold for nms
#    :return:
#    '''
#    if len(score_map.shape) == 4:
#        score_map = score_map[0, 0, :, :]
#        geo_map = geo_map[0, :, :, :]
#    # filter the score map
#    print(score_map.max())
#    #print(geo_map.shape)
#    xy_text = np.argwhere(score_map > score_map_thresh)
#    # sort the text boxes via the y axis
#    xy_text = xy_text[np.argsort(xy_text[:, 0])]
#    # restore
#    start = time.time()
#    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[:, xy_text[:, 0], xy_text[:, 1]]) # N*4*2
#    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
#    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
#    boxes[:, :8] = text_box_restored.reshape((-1, 8))
#    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
#    #timer['restore'] = time.time() - start
#    # nms part
#    start = time.time()
#    #boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
#    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
#    #timer['nms'] = time.time() - start
#
#    if boxes.shape[0] == 0:
#        return None
#    print('{} text boxes after nms'.format(boxes.shape[0]))
#    # here we filter some low score boxes by the average score map, this is different from the orginal paper
#    for i, box in enumerate(boxes):
#        mask = np.zeros_like(score_map, dtype=np.uint8)
#        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
#        boxes[i, 8] = cv2.mean(score_map, mask)[0]
#    boxes = boxes[boxes[:, 8] > box_thresh]
#
#    return boxes


def infer():

    image_shape = [3, 512, 512]
    model = model_builder.EAST(add_conv_body_func=resnet.ResNet(), mode='infer')
    f_geo, f_score = model.build_model(image_shape)
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if not os.path.exists(cfg.pretrained_model):
        raise ValueError("Model path [%s] does not exist." % (cfg.pretrained_model))

    def if_exist(var):
        return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
    fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    infer_reader = reader.infer(cfg.image_path)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    print(model.feeds())
    fetch_list = [f_geo, f_score]
    image_path = "ICDAR2015/ch4_test_images/"
    image_names = os.listdir(image_path)
    image_names.sort()
    print(image_names)
    for iter_id, data in enumerate(infer_reader()):
        #print(image_names[i])
        #data = next(infer_reader())
        im_info = data[0][1]
        print(im_info.shape)     
        ratio_w = im_info[0][3]
        ratio_h = im_info[0][2]
        result = exe.run(fetch_list=[v.name for v in fetch_list],
                         feed=feeder.feed(data),
                         return_numpy=True)
        pred_boxes_v = result[0]
        score =  result[0]
        geometry = result[1]
        boxes = detect(score_map=score, geo_map=geometry)
        #draw_bounding_box_on_image(cfg.image_path, nmsed_out, cfg.draw_threshold,
        #                           labels_map, image)
        f = open("result/res_" + image_names[iter_id].split('.')[0] + ".txt", 'w')
           
#            for k in boxes:
#                print(k)
#                f.write(str(int(k[0]/ratio_h)) + "," + str(int(k[1]/ratio_w)) + "," + str(int(k[2]/ratio_h)) + "," + str(int(k[3]/ratio_w)) + "," + str(int(k[4]/ratio_h)) \
#                        + "," + str(int(k[5]/ ratio_w)) + "," + str(int(k[6]/ratio_h)) + "," + str(int(k[7]/ratio_w)) + "\n") 
        try:
            if boxes.shape[0] != 0:
                boxes = boxes[:, :8].reshape(-1, 4, 2)
                boxes_w = boxes[:, :, 0:1] / ratio_w
                boxes_h = boxes[:, :, 1:] / ratio_h
                boxes = np.concatenate((boxes_w, boxes_h), axis=-1)
                print(boxes.shape)
                for box in boxes:
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    f.write(str(int(box[0][0])) + "," + str(int(box[0][1])) + "," + str(int(box[1][0])) + "," + str(int(box[1][1])) + "," + \
                            str(int(box[2][0])) + "," + str(int(box[2][1])) + "," + str(int(box[3][0])) + "," + str(int(box[3][1])) + "\n")
                print(boxes.shape)
                image = cv2.imread("ICDAR2015/ch4_test_images/" + image_names[iter_id])
                #image = cv2.resize(image, (int(im_info[0][1]), int(im_info[0][0])))
                for i in range(boxes.shape[0]):
                    cv2.polylines(image, [boxes[i].astype('int32')], True, color=(255, 255, 0), thickness=1)
                #image = cv2.resize(image, (int(im_info[0][1]/im_info[0][3]), int(im_info[0][0]/im_info[0][2])))
                print("saving result image", image_names[iter_id])
                cv2.imwrite("./" + image_names[iter_id], image)
        except:
            continue   
if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    check_gpu(args.use_gpu)
    infer()
