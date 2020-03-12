# Author: Tao Hu <taohu620@gmail.com>

# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import matplotlib.pyplot as plt

#dir_name = "../log/deeplab_vgg16.log"

#dir_name = "../log/cs_blur_2.log"
dir_name = "baseline.log"


def draw_deeplab_strong_accuracy():
    f = open(dir_name, "r");
    iter_ind_list = []
    mIoUs = []
    lines = []
    iters = []

    for line in f.readlines():
        line = line.strip("\n")

        if line.find("@res101.slim.2branch.speedup.mcontext.240k.forfigure.py:254][0m mIoU:") != -1:
            print line
            lines.append(line)

    for idx, line in enumerate(lines):
        wordlist = line.split(" ")
        print wordlist
        mIoUs.append(float(wordlist[-1]))
        iters.append(idx)

       

    ax = plt.gca()
    ax.set_title('deeplab_strong Accuracy')
    ax.axis([0, 50, 0, 1])
    plt.plot(iters, mIoUs, '-o', ms=3, lw=0.01, alpha=0.7, mfc='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()


draw_deeplab_strong_accuracy()