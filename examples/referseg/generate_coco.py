# Author: Tao Hu <taohu620@gmail.com>

from pycocotools.coco import COCO
from pycocotools import mask
from tensorpack.utils.segmentation.segmentation import  visualize_label
import numpy as np



coco_dataset = "/data2/dataset/coco"
detection_json_train = "/data2/dataset/annotations/instances_train2014.json"
detection_json_val = "/data2/dataset/annotations/instances_val2014.json"

caption_json_train = "/data2/dataset/annotations/captions_train2014.json"
caption_json_val = "/data2/dataset/annotations/captions_val2014.json"
train_dir = "/data2/dataset/coco/train2014"
val_dir = "/data2/dataset/coco/val2014"




def draw_gt(_coco,img_id):
    img = _coco.loadImgs(img_id)[0]
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    for annId in annIds:
        ann = _coco.loadAnns(annId)[0]

        # polygon
        if type(ann['segmentation']) == list:
            for _instance in ann['segmentation']:
                rle = mask.frPyObjects([_instance], img['height'], img['width'])
        # mask
        else:  # mostly is aeroplane
            if type(ann['segmentation']['counts']) == list:
                rle = mask.frPyObjects([ann['segmentation']], img['height'], img['width'])
            else:
                rle = [ann['segmentation']]
        m = mask.decode(rle)
        img_mask[np.where(m == 1)] = 0000#TODO

import skimage.io as io
import matplotlib.pyplot as plt
#%matplotlib inline

# get all images containing given categories, select one at random
coco_instance = COCO(detection_json_val)
coco_caps = COCO(caption_json_val)


instance_set = set(coco_instance.imgs.keys())
caption_set = set(coco_caps.imgs.keys())

common_set = instance_set & caption_set
for check_img_id in list(common_set):
    print("*" * 40)
    imgIds = coco_instance.getImgIds(imgIds=[check_img_id])
    img = coco_instance.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

    I = io.imread("/data2/dataset/coco/val2014/COCO_val2014_000000{}.jpg".format("%06d" % check_img_id))
    print "/data2/dataset/coco/val2014/COCO_val2014_000000{}.jpg".format("%06d" % check_img_id)

    plt.imshow(I);plt.axis('off')
    annIds = coco_instance.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco_instance.loadAnns(annIds)
    coco_instance.showAnns(anns)
    plt.show()


    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    print coco_caps.showAnns(anns)




