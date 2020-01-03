# coding:utf-8
import glob
import csv
import cv2
import time
import os
import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

#import tensorflow as tf

from data_util import GeneratorEnqueuer
from config import cfg

training_data_path = "icdar2015"
max_image_large_side = 1280
max_text_size = 800
min_text_size = 10
min_crop_side_ratio = 0.1
geometry = 'RBOX'


class ICDAR2015Dataset(object):

    def __init__(self, mode='train'):
        self.training_data_path=training_data_path
        self.max_image_large_side = max_image_large_side
        self. max_text_size = max_text_size
        self.min_text_size = min_text_size
        self.min_crop_side_ratio = min_crop_side_ratio
        self.geometry = geometry
        self.mode = mode
    def get_images(self):
        files = []
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            files.extend(glob.glob(
                os.path.join(self.training_data_path, '*.{}'.format(ext))))
        return files
    
    
    def load_annoataion(self, p):
        '''
        load annotation from the text file
        :param p:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(p):
            return np.array(text_polys, dtype=np.float32)
        with open(p, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
    
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                if label == '*' or label == '###':
                    text_tags.append(True)
                else:
                    text_tags.append(False)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)
    
    
    def polygon_area(self, poly):
        '''
        compute area of a polygon
        :param poly:
        :return:
        '''
        edge = [
            (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
            (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
            (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
            (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
        return np.sum(edge)/2.
    

    def resize_image(self, im, max_side_len=2400):
        '''
        resize image to a size multiple of 32 which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        '''
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        resize_h = max(32, resize_h)
        resize_w = max(32, resize_w)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        #im = cv2.resize(im, (512, 512))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)

    
    def calculate_distance(self, c1, c2):
        return math.sqrt(math.pow(c1[0]-c2[0], 2) + math.pow(c1[1]-c2[1], 2))
    
    
    def choose_best_begin_point(self, pre_result):
        """
        find top-left vertice and resort
        """
        final_result = []
        for coordinate in pre_result:
            x1 = coordinate[0][0]
            y1 = coordinate[0][1]
            x2 = coordinate[1][0]
            y2 = coordinate[1][1]
            x3 = coordinate[2][0]
            y3 = coordinate[2][1]
            x4 = coordinate[3][0]
            y4 = coordinate[3][1]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
            xmax = max(x1, x2, x3, x4)
            ymax = max(y1, y2, y3, y4)
            combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                         [[x2, y2], [x3, y3], [x4, y4], [x1, y1]], 
                         [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], 
                         [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
            dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            force = 100000000.0
            force_flag = 0
            for i in range(4):
                temp_force = self.calculate_distance(combinate[i][0], dst_coordinate[0]) + self.calculate_distance(combinate[i][1], dst_coordinate[1]) + self.calculate_distance(combinate[i][2], dst_coordinate[2]) + self.calculate_distance(combinate[i][3], dst_coordinate[3])
                if temp_force < force:
                    force = temp_force
                    force_flag = i
            #if force_flag != 0:
            #    print("choose one direction!")
            final_result.append(combinate[force_flag])
            
        return final_result
    
    
    def check_and_validate_polys(self, polys, tags, xxx_todo_changeme):
        '''
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        '''
        (h, w) = xxx_todo_changeme
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)
    
        validated_polys = []
        validated_tags = []
    
        polys = self.choose_best_begin_point(polys)   
    
        for poly, tag in zip(polys, tags):
            p_area = self.polygon_area(poly)
            if abs(p_area) < 1:
                # print poly
                print('invalid poly')
                continue
            if p_area > 0:
                print('poly in wrong direction')
                poly = poly[(0, 3, 2, 1), :]
            validated_polys.append(poly)
            validated_tags.append(tag)
        return np.array(validated_polys), np.array(validated_tags)
    
    
    def crop_area(self, im, polys, tags, crop_background=False, max_tries=50):
        '''
        make random crop from the input image
        :param im:
        :param polys:
        :param tags:
        :param crop_background:
        :param max_tries:
        :return:
        '''
        h, w, _ = im.shape
        pad_h = h//10
        pad_w = w//10
        h_array = np.zeros((h + pad_h*2), dtype=np.int32)
        w_array = np.zeros((w + pad_w*2), dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx+pad_w:maxx+pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny+pad_h:maxy+pad_h] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return im, polys, tags
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w-1)
            xmax = np.clip(xmax, 0, w-1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h-1)
            ymax = np.clip(ymax, 0, h-1)
            if xmax - xmin < self.min_crop_side_ratio*w or ymax - ymin < self.min_crop_side_ratio*h:
                # area too small
                continue
            if polys.shape[0] != 0:
                poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                    & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # no text in this area
                if crop_background:
                    return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
                else:
                    continue
            im = im[ymin:ymax+1, xmin:xmax+1, :]
            polys = polys[selected_polys]
            tags = tags[selected_polys]
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return im, polys, tags
    
        return im, polys, tags
    
    
    def shrink_poly(self, poly, r):
        '''
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param r: r in the paper
        :return: the shrinked poly
        '''
        # shrink ratio
        R = 0.3
        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                        np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly
    
    
    def point_dist_to_line(self, p1, p2, p3):
        # compute the distance from p3 to p1-p2
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    
    
    def fit_line(self, p1, p2):
        # fit a line ax+by+c = 0
        if p1[0] == p1[1]:
            return [1., 0., -p1[0]]
        else:
            [k, b] = np.polyfit(p1, p2, deg=1)
            return [k, -1., b]
    
    
    def line_cross_point(self, line1, line2):
        # line1 0= ax+by+c, compute the cross point of line1 and line2
        if line1[0] != 0 and line1[0] == line2[0]:
            print('Cross point does not exist')
            return None
        if line1[0] == 0 and line2[0] == 0:
            print('Cross point does not exist')
            return None
        if line1[1] == 0:
            x = -line1[2]
            y = line2[0] * x + line2[2]
        elif line2[1] == 0:
            x = -line2[2]
            y = line1[0] * x + line1[2]
        else:
            k1, _, b1 = line1
            k2, _, b2 = line2
            x = -(b1-b2)/(k1-k2)
            y = k1*x + b1
        return np.array([x, y], dtype=np.float32)
    
    
    def line_verticle(self, line, point):
        # get the verticle line from line across point
        if line[1] == 0:
            verticle = [0, -1, point[1]]
        else:
            if line[0] == 0:
                verticle = [1, 0, -point[0]]
            else:
                verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
        return verticle
    
    
    def rectangle_from_parallelogram(self, poly):
        '''
        fit a rectangle from a parallelogram
        :param poly:
        :return:
        '''
        p0, p1, p2, p3 = poly
        angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
        if angle_p0 < 0.5 * np.pi:
            if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
                # p0 and p2
                ## p0
                p2p3 = self.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.line_verticle(p2p3, p0)
    
                new_p3 = self.line_cross_point(p2p3, p2p3_verticle)
                ## p2
                p0p1 = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.line_verticle(p0p1, p2)
    
                new_p1 = self.line_cross_point(p0p1, p0p1_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
            else:
                p1p2 = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.line_verticle(p1p2, p0)
    
                new_p1 = self.line_cross_point(p1p2, p1p2_verticle)
                p0p3 = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.line_verticle(p0p3, p2)
    
                new_p3 = self.line_cross_point(p0p3, p0p3_verticle)
                return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
                # p1 and p3
                ## p1
                p2p3 = self.fit_line([p2[0], p3[0]], [p2[1], p3[1]])
                p2p3_verticle = self.line_verticle(p2p3, p1)
    
                new_p2 = self.line_cross_point(p2p3, p2p3_verticle)
                ## p3
                p0p1 = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                p0p1_verticle = self.line_verticle(p0p1, p3)
    
                new_p0 = self.line_cross_point(p0p1, p0p1_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
            else:
                p0p3 = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                p0p3_verticle = self.line_verticle(p0p3, p1)
    
                new_p0 = self.line_cross_point(p0p3, p0p3_verticle)
                p1p2 = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                p1p2_verticle = self.line_verticle(p1p2, p3)
    
                new_p2 = self.line_cross_point(p1p2, p1p2_verticle)
                return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
    
    
    def sort_rectangle(self, poly):
        # sort the four coordinates of the polygon, points in poly should be sorted clockwise
        # First find the lowest point
        p_lowest = np.argmax(poly[:, 1])
        if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
            # 底边平行于X轴, 那么p0为左上角 - if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
            p0_index = np.argmin(np.sum(poly, axis=1))
            p1_index = (p0_index + 1) % 4
            p2_index = (p0_index + 2) % 4
            p3_index = (p0_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
        else:
            # 找到最低点右边的点 - find the point that sits right to the lowest point
            p_lowest_right = (p_lowest - 1) % 4
            p_lowest_left = (p_lowest + 1) % 4
            angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
            # assert angle > 0
            if angle <= 0:
                print(angle, poly[p_lowest], poly[p_lowest_right])
            if angle/np.pi * 180 > 45:
                # 这个点为p2 - this point is p2
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
            else:
                # 这个点为p3 - this point is p3
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle
    
    
    def generate_rbox(self, im_size, polys, tags):
        h, w = im_size
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)
        # mask used during traning, to ignore some hard areas
        training_mask = np.ones((h, w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]
    
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = self.shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < self.min_text_size:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
    
            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
            # if geometry == 'RBOX':
            # 对任意两个顶点的组合生成一个平行四边形 - generate a parallelogram for any combination of two vertices
            fitted_parallelograms = []
            for i in range(4):
                p0 = poly[i]
                p1 = poly[(i + 1) % 4]
                p2 = poly[(i + 2) % 4]
                p3 = poly[(i + 3) % 4]
                edge = self.fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                backward_edge = self.fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                forward_edge = self.fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                if self.point_dist_to_line(p0, p1, p2) > self.point_dist_to_line(p0, p1, p3):
                    # 平行线经过p2 - parallel lines through p2
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p2[0]]
                    else:
                        edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
                else:
                    # 经过p3 - after p3
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p3[0]]
                    else:
                        edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
                # move forward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p2 = self.line_cross_point(forward_edge, edge_opposite)
                if self.point_dist_to_line(p1, new_p2, p0) > self.point_dist_to_line(p1, new_p2, p3):
                    # across p0
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p0[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
                else:
                    # across p3
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p3[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
                new_p0 = self.line_cross_point(forward_opposite, edge)
                new_p3 = self.line_cross_point(forward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
                # or move backward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p3 = self.line_cross_point(backward_edge, edge_opposite)
                if self.point_dist_to_line(p0, p3, p1) > self.point_dist_to_line(p0, p3, p2):
                    # across p1
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p1[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
                else:
                    # across p2
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p2[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
                new_p1 = self.line_cross_point(backward_opposite, edge)
                new_p2 = self.line_cross_point(backward_opposite, edge_opposite)
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            areas = [Polygon(t).area for t in fitted_parallelograms]
            parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
            # sort thie polygon
            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]
    
            rectange = self.rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = self.sort_rectangle(rectange)
    
            p0_rect, p1_rect, p2_rect, p3_rect = rectange
            for y, x in xy_in_poly:
                point = np.array([x, y], dtype=np.float32)
                # top
                geo_map[y, x, 0] = self.point_dist_to_line(p0_rect, p1_rect, point)
                # right
                geo_map[y, x, 1] = self.point_dist_to_line(p1_rect, p2_rect, point)
                # down
                geo_map[y, x, 2] = self.point_dist_to_line(p2_rect, p3_rect, point)
                # left
                geo_map[y, x, 3] = self.point_dist_to_line(p3_rect, p0_rect, point)
                # angle
                geo_map[y, x, 4] = rotate_angle
        return score_map, geo_map, training_mask
    
    def ger_roidb(self):
        im_infos = []
        image_list = np.array(self.get_images())
        image_list = np.array(self.get_images())
        print('{} training images in {}'.format(
            image_list.shape[0], self.training_data_path))
        index = np.arange(0, image_list.shape[0])
        count = 0
        for i in index:
            im_fn = image_list[i]
            im = cv2.imread(im_fn)
            h, w, _ = im.shape
            txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
            if not os.path.exists(txt_fn):
                print('text file {} does not exists'.format(txt_fn))
                continue
            text_polys, text_tags = self.load_annoataion(txt_fn)
            gt_boxes, difficult = self.check_and_validate_polys(text_polys, text_tags, (h, w))
            difficult = np.array(difficult)
            gt_classes = np.ones_like(difficult)
            count += 1
            im_info = {
                'im_id': count,
                'gt_classes': gt_classes,
                'image':im_fn,
                'boxes': gt_boxes,
                'height': h,
                'width': w,
                'is_difficult': is_difficult
            }
            im_infos.append(im_info)   
        return im_infos
    
    def generator(self, input_size=512, batch_size=24,
                  background_ratio=3./8,
                  random_scale=np.array([0.5, 1, 2.0, 3.0]),
                  vis=False):
        image_list = np.array(self.get_images())
        print('{} training images in {}'.format(
            image_list.shape[0], self.training_data_path))
        index = np.arange(0, image_list.shape[0])
        while True:
            np.random.shuffle(index)
            images = []
            image_fns = []
            score_maps = []
            geo_maps = []
            training_masks = []
            for i in index:
                try:
                    im_fn = image_list[i]
                    im = cv2.imread(im_fn)
                    #im = im.astype(np.float32, copy=False)
                    #im = im[:, :, ::-1].astype(np.float32)                
                    #im = im / 255.0
                    #im -= cfg.pixel_means
                    #im /= cfg.std
                    #im = im[:, :, ::-1].astype(np.float32)
                    # print im_fn
                    h, w, _ = im.shape
                    txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                    if not os.path.exists(txt_fn):
                        print('text file {} does not exists'.format(txt_fn))
                        continue
    
                    text_polys, text_tags = self.load_annoataion(txt_fn)
                     
                    text_polys, text_tags = self.check_and_validate_polys(text_polys, text_tags, (h, w))
                    # if text_polys.shape[0] == 0:
                    #     continue
                    # random scale this image
                    if self.mode == 'train':
                        rd_scale = np.random.choice(random_scale)
                        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                        text_polys *= rd_scale
                        # print rd_scale
                        # random crop a area from image
                        if np.random.rand() < background_ratio:
                            # crop background
                            im, text_polys, text_tags = self.crop_area(im, text_polys, text_tags, crop_background=True)
                            if text_polys.shape[0] > 0:
                                # cannot find background
                                continue
                            # pad and resize image
                            new_h, new_w, _ = im.shape
                            max_h_w_i = np.max([new_h, new_w, input_size])
                            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                            im_padded[:new_h, :new_w, :] = im.copy()
                            im = cv2.resize(im_padded, dsize=(input_size, input_size))
                            score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                            geo_map_channels = 5 if self.geometry == 'RBOX' else 8
                            geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                            training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                        else:
                            im, text_polys, text_tags = self.crop_area(im, text_polys, text_tags, crop_background=False)
                            if text_polys.shape[0] == 0:
                                continue
                            h, w, _ = im.shape
        
                            # pad the image to the training input size or the longer side of image
                            new_h, new_w, _ = im.shape
                            max_h_w_i = np.max([new_h, new_w, input_size])
                            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                            im_padded[:new_h, :new_w, :] = im.copy()
                            im = im_padded
                            # resize the image to input size
                            new_h, new_w, _ = im.shape
                            resize_h = input_size
                            resize_w = input_size
                            im = cv2.resize(im, dsize=(resize_w, resize_h))
                            resize_ratio_3_x = resize_w/float(new_w)
                            resize_ratio_3_y = resize_h/float(new_h)
                            text_polys[:, :, 0] *= resize_ratio_3_x
                            text_polys[:, :, 1] *= resize_ratio_3_y
                            new_h, new_w, _ = im.shape
                            score_map, geo_map, training_mask = self.generate_rbox((new_h, new_w), text_polys, text_tags)
    
                    im = im[:, :, ::-1].astype(np.float32)
                    im = im / 255.0
                    im -= cfg.pixel_means
                    im /= cfg.std
    
                    im = im.transpose((2, 0, 1))
                    im = np.expand_dims(im, axis=0)
                    images.append(im)
                    image_fns.append(im_fn)
                    score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
                    score_map = score_map.transpose((2, 0, 1))
                    score_map = np.expand_dims(score_map, axis=0)
                    score_maps.append(score_map)
                    geo_map = geo_map[::4, ::4, :].astype(np.float32)
                    geo_map = geo_map.transpose((2, 0, 1))
                    geo_map = np.expand_dims(geo_map, axis=0)
                    geo_maps.append(geo_map)
                    training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)
                    training_mask = training_mask.transpose((2, 0, 1))
                    training_mask = np.expand_dims(training_mask, axis=0)
                    training_masks.append(training_mask)
    
                    if len(images) == batch_size:
                        yield np.concatenate(images), image_fns, np.concatenate(score_maps), np.concatenate(geo_maps), np.concatenate(training_masks)
                        images = []
                        image_fns = []
                        score_maps = []
                        geo_maps = []
                        training_masks = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
    
    
    def get_batch(self, num_workers, **kwargs):
        try:
            enqueuer = GeneratorEnqueuer(self.generator(**kwargs), use_multiprocessing=True)
            print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
            enqueuer.start(max_queue_size=10, workers=num_workers)
            generator_output = None
            while True:
                while enqueuer.is_running():
                    if not enqueuer.queue.empty():
                        generator_output = enqueuer.queue.get()
                        break
                    else:
                        time.sleep(0.01)
                yield generator_output
                generator_output = None
        finally:
            if enqueuer is not None:
                enqueuer.stop()



if __name__ == '__main__':
    pass
