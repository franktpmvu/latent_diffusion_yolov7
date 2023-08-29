# coding: utf-8
import os
import numpy as np
import re
from os.path import join, abspath
import cv2

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


if __name__ == '__main__':

    txt_path = './label.txt'
    folder_path = './jpeg/'

    output = ''
    output_list = []
    count = 0
    scale = 416
    txt_file = open(txt_path)
    txt_file.readline()
    for line in txt_file:
        try:
            line = line.strip()
            lines = line.split(' ')
            name, p1, p2, p3, p4 = lines

            p1 = np.array(p1.split(',')).astype(np.int)
            p2 = np.array(p2.split(',')).astype(np.int)
            p3 = np.array(p3.split(',')).astype(np.int)
            p4 = np.array(p4.split(',')).astype(np.int)

            p1x = p1[0]
            p1y = p1[1]
            p2x = p2[0]
            p2y = p2[1]
            p3x = p3[0]
            p3y = p3[1]
            p4x = p4[0]
            p4y = p4[1]
            cx = (p1x + p2x + p3x + p4x)//4
            cy = (p1y + p2y + p3y + p4y)//4

            full_path = abspath(folder_path)
            full_path = join(full_path,name)

            # Rotate on XY-plane
            # v = np.array([p2[0]-p1[0], p2[1]-p1[1]])
            # unit_x = np.array([1,0])  # X-axis
            # theta = angle_between(v, unit_x)  # compute angle between eyes and X-axis
            # theta = round(theta, 3)
            # if ((p2[0]-p1[0])*(p2[1]-p1[1])) > 0 :  # Counterclockwise rotation
            #     theta = -theta

            # # Find x-axis shearing
            # sin = np.sin(theta)
            # cos = np.cos(theta)
            # trans_matrix = np.float32(np.array([[cos, -sin], [sin, cos]]))
            # new_p1 = np.dot(trans_matrix,p1)
            # new_p3 = np.dot(trans_matrix,p3)
            # shearing = (new_p1[0] - new_p3[0])/(new_p3[1] - new_p1[1])
            # shearing = round(shearing,5)

            # # Find perspective value
            # pts1 = np.float32([p1,p2,p3,p4])
            # pts2 = np.float32([[0,0],[100,0],[0,50],[100,50]])
            # M = cv2.getPerspectiveTransform(pts1,pts2)
            # a20 = round(M[2][0],8)
            # a21 = round(M[2][1],8)

            # x = min(p1[0],p3[0])
            # y = min(p1[1],p2[1])
            # w = max(p2[0],p4[0]) - x
            # h = max(p3[1],p4[1]) - y
            write_msg = '%s %r,%r,%r,%r,%r,%r,%r,%r,%r,%r,0' % (full_path, cx, cy, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
            output = output + write_msg + '\n'
        except Exception as e:
            print(e)

        count += 1
        if count%2000 == 0:
            print(count)
            output_list.append(output)
            output = ''
            break

    output_list.append(output)
    output = ''
    for list_str in output_list:
        output += list_str

    fo = open("train.txt", "w")
    fo.write(output)
    fo.close()

