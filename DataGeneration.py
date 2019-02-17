### This code sample is provided by Mez Gebre's repository "deep_homography_estimation"
#   https://github.com/mez/deep_homography_estimation
#   Dataset_Generation_Visualization.ipynb

from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv

img_path = 'C://Users//MEL//Desktop//14572634_1249839005066331_1506895333_o.jpg'
img = cv2.imread(img_path,0)
img = cv2.resize(img,(320,240))

rho          = 32
patch_size   = 128
top_point    = (32,32)
left_point   = (patch_size+32, 32)
bottom_point = (patch_size+32, patch_size+32)
right_point  = (32, patch_size+32)
test_image = img.copy()
four_points = [top_point, left_point, bottom_point, right_point]

perturbed_four_points = []
for point in four_points:
    perturbed_four_points.append((point[0] + random.randint(-rho,rho), point[1]+random.randint(-rho,rho)))

H = cv2.getPerspectiveTransform( np.float32(four_points), np.float32(perturbed_four_points) )
H_inverse = inv(H)

warped_image = cv2.warpPerspective(img,H_inverse, (320,240))

annotated_warp_image = warped_image.copy()

Ip1 = annotated_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]
Ip2 = warped_image[top_point[1]:bottom_point[1],top_point[0]:bottom_point[0]]

training_image = np.dstack((Ip1, Ip2))
H_four_points = np.subtract(np.array(perturbed_four_points), np.array(four_points))
datum = (training_image, H_four_points)

