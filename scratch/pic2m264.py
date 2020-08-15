import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('D:/git/spacex_case/figures/throughput*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('SpaceX_throughput.avi',cv2.VideoWriter_fourcc('M','P','E','G'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
