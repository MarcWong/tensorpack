# Author: Tao Hu <taohu620@gmail.com>
train_images = "/data1/dataset/AerialImage/train/images"
train_gt = "/data1/dataset/AerialImage/train/gt"

val_images = "/data1/dataset/AerialImage/val/images"
val_gt = "/data1/dataset/AerialImage/val/gt"

expand_train_gt ="/data1/dataset/AerialImage_Expand/train/gt"
expand_val_gt ="/data1/dataset/AerialImage_Expand/val/gt"

import shutil,os,cv2
from tqdm import tqdm
import numpy as np

#os.makedirs(expand_train_gt)
#os.makedirs(expand_val_gt)

print "train gt.."
for filename in tqdm(os.listdir(train_gt)):
    img = cv2.imread(os.path.join(train_gt,filename),0)
    img = img*255
    edge = cv2.Canny(img,100,200)
    edge = edge/255
    cv2.imwrite(os.path.join(expand_train_gt,filename),edge)

print "val gt"
for filename in tqdm(os.listdir(val_gt)):
    img = cv2.imread(os.path.join(val_gt,filename),0)
    img = img * 255
    edge = cv2.Canny(img,100,200)
    edge = edge/255
    cv2.imwrite(os.path.join(expand_val_gt,filename),edge)




