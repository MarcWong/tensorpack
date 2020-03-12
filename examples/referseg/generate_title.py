# Author: Tao Hu <taohu620@gmail.com>
import os,cv2
import numpy as np
from tqdm import tqdm

from tensorpack.utils.segmentation.segmentation import id_to_name,visualize_label

pascal_img_path = "/home/hutao/dataset/pascalvoc2012/normal_voc2012/val"
pascal_gt_path = "/home/hutao/dataset/pascalvoc2012/normal_voc2012/val_anno"
pascal_new_img_path = "/home/hutao/dataset/pascalvoc2012/normal_voc2012/newval"


for file in tqdm(os.listdir(pascal_img_path)):
    file_name, extension = file.split(".")
    img = cv2.imread(os.path.join(pascal_img_path, file))
    gt_filename = "{}.png".format(file_name)
    gt_img = cv2.imread(os.path.join(pascal_gt_path, gt_filename),cv2.IMREAD_GRAYSCALE)
    classes = np.unique(gt_img)
    class_names = []
    shape = classes.shape
    for i in range(shape[0]):
        if classes[i] == 255 or classes[i]==0:
            continue
        class_names.append(id_to_name[classes[i]])

    new_name = "{}_{}.{}".format(file_name,"_".join(class_names),"jpg")
    cv2.imwrite(os.path.join(pascal_new_img_path,new_name),np.concatenate([img,visualize_label(gt_img)],axis=1))







