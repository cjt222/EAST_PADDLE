#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import os
import numpy as np
import paddle.fluid as fluid
import math
from config import cfg
import six
import numpy as np
import cv2
import Polygon as plg
import lanms
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from config import cfg
from icdar import restore_rectangle
import logging
logger = logging.getLogger(__name__)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, 0, :, :]
        geo_map = geo_map[0, :, :, :]
    # filter the score map
    print(score_map.max())
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[:, xy_text[:, 0], xy_text[:, 1]]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    #boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None
    print('{} text boxes after nms'.format(boxes.shape[0]))
    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def get_key_dict(out, data, key):
    res = {}
    for i in range(len(key)):
        if i == 0:
            res[key[i]] = out
        else:
            res[key[i]] = data[i]
    return res


def get_labels_maps():
    default_labels_maps = {1: 'text'}

    return default_labels_maps


def draw_bounding_box_on_image(image_path,
                               image_name,
                               nms_out,
                               im_scale,
                               draw_threshold=0.8):
    #if image is None:
    image = Image.open(os.path.join(image_path, image_name))
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size

    labels_map = get_labels_maps()
    for dt in np.array(nms_out):
        num_id, score = dt.tolist()[:2]
        x1, y1, x2, y2, x3, y3, x4, y4 = dt.tolist()[2:] / im_scale
        if score < draw_threshold:
            continue
        draw.line(
            [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
            width=2,
            fill='red')
        if image.mode == 'RGB':
            draw.text((x1, y1), labels_map[num_id], (255, 255, 0))
    print("image with bbox drawed saved as {}".format(image_name))
    image.save(image_name)


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return plg.Polygon(point_mat)


def clip_box(bbox, im_info):

    h = im_info[0]
    w = im_info[1]
    res = []
    for b in bbox:
        pts = b.reshape(4, 2)
        pts[np.where(pts < 0)] = 1
        pts[np.where(pts[:, 0] > w), 0] = w - 1
        pts[np.where(pts[:, 1] > h), 1] = h - 1
        pts = pts.reshape(-1)
        pts /= im_info[2]
        res.append(pts)

    return np.array(res)


def get_union(det, gt):
    area_det = det.area()
    area_gt = gt.area()
    return area_det + area_gt - get_intersection(det, gt)


def get_intersection_over_union(det, gt):
    try:
        return get_intersection(det, gt) / get_union(det, gt)
    except:
        return 0


def get_intersection(det, gt):
    inter = det & gt
    if len(inter) == 0:
        return 0
    return inter.area()


def icdar_box_eval(result, thresh):

    matched_sum = 0
    num_global_care_gt = 0
    num_global_care_det = 0
    for res in result:
        im_info = res['im_info']
        h = im_info[1]
        w = im_info[2]
        gt_boxes = res['gt_box']
        pred_boxes = res['bbox']
        pred_boxes = pred_boxes[np.where(pred_boxes[:, 1] > thresh)]
        pred_boxes = pred_boxes[:, 2:]
        pred_boxes = clip_box(pred_boxes, im_info)

        is_difficult = res['is_difficult']
        det_matched = 0

        iou_mat = np.empty([1, 1])

        gt_pols = []
        det_pols = []

        gt_pol_points = []
        det_pol_points = []

        gt_dont_care_pols_num = []
        det_dont_care_pols_num = []

        det_matched_nums = []

        points_list = list(gt_boxes)

        dony_care = is_difficult.reshape(-1)
        for i, points in enumerate(points_list):
            gt_pol = polygon_from_points(list(points))
            gt_pols.append(gt_pol)
            gt_pol_points.append(list(points))
            if dony_care[i] == 1:
                gt_dont_care_pols_num.append(len(gt_pols) - 1)
        for i, points in enumerate(pred_boxes):
            points = list(points.reshape(8).astype(np.int32))
            det_pol = polygon_from_points(points)
            det_pols.append(det_pol)
            det_pol_points.append(points)
            if len(gt_dont_care_pols_num) > 0:
                for dont_care_pol in gt_dont_care_pols_num:
                    dont_care_pol = gt_pols[dont_care_pol]
                    intersected_area = get_intersection(dont_care_pol, det_pol)
                    pd_dimensions = det_pol.area()
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                    if (precision > 0.5):
                        det_dont_care_pols_num.append(len(det_pols) - 1)
                        break
        if len(gt_pols) > 0 and len(det_pols) > 0:
            # Calculate IoU and precision matrixs
            output_shape = [len(gt_pols), len(det_pols)]
            iou_mat = np.empty(output_shape)
            gt_rect_mat = np.zeros(len(gt_pols), np.int8)
            det_rect_mat = np.zeros(len(det_pols), np.int8)
            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    p_d = gt_pols[gt_num]
                    p_g = det_pols[det_num]
                    iou_mat[gt_num, det_num] = get_intersection_over_union(p_d,
                                                                           p_g)

            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    if gt_rect_mat[gt_num] == 0 and det_rect_mat[
                            det_num] == 0 and gt_num not in gt_dont_care_pols_num and det_num not in det_dont_care_pols_num:
                        if iou_mat[gt_num, det_num] > 0.5:
                            gt_rect_mat[gt_num] = 1
                            det_rect_mat[det_num] = 1
                            det_matched += 1
                            det_matched_nums.append(det_num)
        num_gt_care = (len(gt_pols) - len(gt_dont_care_pols_num))
        num_det_care = (len(det_pols) - len(det_dont_care_pols_num))
        matched_sum += det_matched
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care
    method_recall = 0 if num_global_care_gt == 0 else float(
        matched_sum) / num_global_care_gt
    method_precision = 0 if num_global_care_det == 0 else float(
        matched_sum) / num_global_care_det
    method_hmean = 0 if method_recall + method_precision == 0 else 2 * method_recall * method_precision / (
        method_recall + method_precision)
    logger.info('Recall {}'.format(method_recall))
    logger.info('Precision {}'.format(method_precision))
    logger.info('F1 {}'.format(method_hmean))


def icdar_eval(result):
    icdar_box_eval(result, 0.8)
