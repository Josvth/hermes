import cv2
import numpy as np
import glob

import argparse
import sys

import cv2
import numpy as np
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Turn stack of pngs into an avi movie.')
parser.add_argument('-d', type=str, help='Directory of PNGs')
parser.add_argument('-n', type=str, default='movie.avi', help='Name of movie')

args = parser.parse_args()

img_array = []
for filename in tqdm(glob.glob(args.d + '/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    #fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
    out = cv2.VideoWriter(args.n, fourcc, frameSize=size, fps=60)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

# img_array = []
# for filename in glob.glob('D:/git/spacex_case/figures/throughput*.png'):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
#
# out = cv2.VideoWriter('SpaceX_throughput.avi',cv2.VideoWriter_fourcc('M','P','E','G'), 1, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()
