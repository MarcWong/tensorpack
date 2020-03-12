# Author: Tao Hu <taohu620@gmail.com>
import os,cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from tensorpack.utils.segmentation.segmentation import id_to_name,visualize_label
from tensorpack.utils.segmentation.coco_util import generate_id2trainid, generate_image_mask

coco_val = "/data2/dataset/coco/train2014"
#coco_val = "/data2/dataset/coco/val2014"

#target_coco_val = "/data2/dataset/coco/new_val2014"
target_coco_val = "/data2/dataset/coco/new_train2014"

detection_json_val = "/data2/dataset/annotations/instances_train2014.json"
#detection_json_val = "/data2/dataset/annotations/instances_val2014.json"



coco_instance = COCO(detection_json_val)


instance_set = set(coco_instance.imgs.keys())


print("instance_set length: {}".format(len(instance_set)))



cat_dict = generate_id2trainid(coco_instance)

for check_img_id in tqdm(list(instance_set)):
    imgIds = coco_instance.getImgIds(imgIds=[check_img_id])
    img = coco_instance.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    file_path = os.path.join(coco_val,coco_instance.imgs[check_img_id]['file_name'])
    I = cv2.imread(file_path)
    print file_path

    annIds = coco_instance.getAnnIds(imgIds=imgIds)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    class_names = set()
    for annId in annIds:
        img_mask, ann = generate_image_mask(coco_instance,img_mask,annId, cat_dict)
        class_names.add(coco_instance.cats[ann['category_id']]['name'])

    class_names = list(class_names)
    target_path = os.path.join(target_coco_val,coco_instance.imgs[check_img_id]['file_name'])\
        .replace(".jpg","_{}.jpg".format("_".join(class_names)))

    cv2.imwrite(target_path,np.concatenate([I,visualize_label(img_mask,class_num=80)],axis=1))















