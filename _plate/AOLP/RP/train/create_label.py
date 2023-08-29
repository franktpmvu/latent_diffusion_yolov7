# coding: utf-8
import os
import numpy as np
import re
from os.path import join, abspath
import cv2


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

            #write_msg = '%s %r,%r,%r,%r,%r,%r,%r,%r,%r,%r,0' % (full_path, cx, cy, p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)
            xmin = min(p1x, p2x, p3x, p4x)
            ymin = min(p1y, p2y, p3y, p4y)
            xmax = max(p1x, p2x, p3x, p4x)
            ymax = max(p1y, p2y, p3y, p4y)
            write_msg = '%s %r,%r,%r,%r,0' % (full_path, xmin, ymin, xmax, ymax)
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

    #fo = open("train.txt", "w")
    fo = open("train2.txt", "w")
    fo.write(output)
    fo.close()

