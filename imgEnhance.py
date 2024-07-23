import cv2
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch

img = cv2.imread('data/ExLPose/dark/imgs_0206_vid000024_exp200_dark_000052__gain_3.60_exposure_2500.png')

rgb_mean_LL = np.mean(img, axis=(0, 1))
scaling_LL = 255*0.4 / rgb_mean_LL
image_LL = img * scaling_LL

cv2.imwrite('output_image.jpg', image_LL)