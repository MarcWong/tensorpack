# Author: Tao Hu <taohu620@gmail.com>
import cv2
from matplotlib import pyplot as plt
import numpy as np

baseline32 = cv2.imread("baseline-fold0-test-32.png")[:,320*3:320*4,1]/255.0
baseline71 = cv2.imread("baseline-fold0-test-71.png")[:,320*3:320*4,1]/255.0
attention32 = cv2.imread("attention-fold0-test-32.png")[:,320*3:320*4,1]/255.0
attention71 = cv2.imread("attention-fold0-test-71.png")[:,320*3:320*4,1]/255.0

import matplotlib.patches as mpatches
plt.xlabel('Activations')
plt.ylabel('Count')
plt.title('Histogram')
before = mpatches.Patch(color='g', label='Before Attention')
after = mpatches.Patch(color='r', label='After Attention')
plt.legend(handles=[before,after])
plt.hist(baseline32.flatten(), bins = np.arange(0,1,0.025).tolist(), facecolor='g', alpha=0.75,label="Before Attention")
plt.hist(attention32.flatten(), bins = np.arange(0,1,0.025).tolist(), facecolor='r', alpha=0.75, label="After Attention")
plt.show()
