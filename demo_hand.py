import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.hand import Hand
from src import util

hand_estimation = Hand('model/hand_pose_model.pth')
hand_image = 'images/custom/Hand.JPG'
source = cv2.imread(hand_image)
canvas = copy.deepcopy(source)
peaks = hand_estimation(source)
peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0])
peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1])
canvas = util.draw_handpose(canvas, [peaks])

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
