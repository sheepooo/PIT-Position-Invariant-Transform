import os
import math
import numpy as np
import argparse
import xml.etree.ElementTree as ET
from PIT_tensor import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='FOV')
    parser.add_argument('--dataset', dest='dataset',
                        help='cityscapes, foggy_cityscapes, kitti, vkitti',
                        default='kitti', type=str)
    parser.add_argument('--fovx', dest='fovx',
                        help='fovx, the unit is degree',
                        default='0', type=int)
    parser.add_argument('--fovy', dest='fovy',
                        help='fovy, the unit is degree',
                        default='0', type=int)
    parser.add_argument('--root_path', dest='root_path',
                        help='the path to place the image folder',
                        default='./', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    for root_folder, dirs, files in os.walk(args.root_path, topdown=True):
        print(root_folder)
        for file_name in files:
            if not file_name.endswith('.xml'):
                continue
            print(file_name)
            file_path = os.path.join(root_folder, file_name)
            tree = ET.parse(file_path)
            root = tree.getroot()

            size = root.find('size')
            width, height = int(size[0].text), int(size[1].text)
            pit = PIT_module(width, height, args.fovx / 180 * math.pi, args.fovy / 180 * math.pi)

            new_width, new_height = pit.coord_plain_to_arc_scalar(width, height)
            new_width, new_height = int(new_width), int(new_height)
            # print(new_width)
            size[0].text, size[1].text = str(new_width), str(new_height)

            for child in root:
              if child.tag == 'object':
                  bbox = child.find('bndbox')
                  xmin, ymin = int(bbox[0].text), int(bbox[1].text)
                  xmax, ymax = int(bbox[2].text), int(bbox[3].text)
                  #print('--',xmin, ymin, xmax, ymax, new_width, new_height)
                  xmin, ymin = pit.coord_plain_to_arc_scalar(xmin, ymin)
                  xmax, ymax = pit.coord_plain_to_arc_scalar(xmax, ymax)
                  xmin, ymin = int(np.around(xmin)), int(np.around(ymin))
                  xmax, ymax = int(np.around(xmax)), int(np.around(ymax))
                  #print(xmin, ymin, xmax, ymax, new_width, new_height)
                  assert xmin >= 0 and ymin >= 0 and xmax - 1 <= new_width and ymax - 1 <= new_height
                  if xmax > new_width:
                      xmax = new_width
                  if ymax > new_height:
                      ymax = new_height
                  bbox[0].text, bbox[1].text = str(xmin), str(ymin)
                  bbox[2].text, bbox[3].text = str(xmax), str(ymax)
            # break
            tree.write(file_path)

        print('All annotation projected')
