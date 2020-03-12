# Author: Tao Hu <taohu620@gmail.com>
# the main code is borrowed from https://github.com/yunjey/show-attend-and-tell/blob/master/prepro.py

import os
import gzip
import numpy as np
import cv2
from tqdm import tqdm

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.utils.segmentation.segmentation import visualize_label
from tensorpack.utils.segmentation.coco_util import generate_id2trainid, generate_image_mask

__all__ = ['DataLoader']

caption_train_json = '/data2/dataset/annotations/captions_train2014.json'
instance_train_json = "/data2/dataset/annotations/instances_train2014.json"
caption_val_json = '/data2/dataset/annotations/captions_val2014.json'
instance_val_json = "/data2/dataset/annotations/instances_val2014.json"

coco_train_dir = "/data2/dataset/coco/train2014"
coco_val_dir = "/data2/dataset/coco/val2014"
class_num = 81


# if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
word_count_threshold = 1

from collections import Counter
import numpy as np
import pandas as pd
import os
import json
from pycocotools.coco import COCO
from pycocotools import mask

def resize_and_pad(im, input_h, input_w, interp):
    # Resize and pad im to input_h x input_w size
    im_h, im_w = im.shape[:2]
    scale = min(input_h*1.0 / im_h, input_w*1.0 / im_w)
    resized_h = int(np.round(im_h * scale))
    resized_w = int(np.round(im_w * scale))
    pad_h = int(np.floor(input_h - resized_h) / 2)
    pad_w = int(np.floor(input_w - resized_w) / 2)

    resized_im = cv2.resize(im, (resized_h, resized_w), interpolation=interp)
    if resized_im.ndim == 2:
        resized_im = resized_im[:,:,np.newaxis]# avoid swallow last dimension in grey image by cv2.resize
    if im.ndim > 2:
        new_im = np.zeros((input_h, input_w, im.shape[2]), dtype=resized_im.dtype)
        new_im[pad_w:pad_w + resized_w, pad_h:pad_h + resized_h, ...] = resized_im  # reverse order
    else:
        new_im = np.zeros((input_h, input_w), dtype=resized_im.dtype)
        new_im[pad_w:pad_w + resized_w, pad_h:pad_h + resized_h] = resized_im  # reverse order


    return new_im

def _process_caption_data(caption_file, max_length, max_image_num = -1):
    coco = COCO(caption_file)
    caption_data = coco.dataset
    img_to_caption = {}
    img_ids = [tmp['id'] for tmp in caption_data['images']]
    for img_id in img_ids:
        annIds = coco.getAnnIds(imgIds=img_id)
        # only consider one caption
        img_to_caption[img_id] = coco.loadAnns(annIds[0])[0]['caption']

    delete_keys = []
    for img_id, caption in tqdm(img_to_caption.items()):
        caption = caption.replace('.', '').replace(',', '').replace("'", "").replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(")", "").replace('-', ' ')
        caption = " ".join(caption.split())  # replace multiple spaces
        caption = caption.lower()
        if len(caption.split(" ")) > max_length:
            delete_keys.append(img_id)
        img_to_caption[img_id] = caption

    # delete captions if size is larger than max_length
    logger.info("The number of captions before deletion: %d" % len(img_to_caption.keys()))
    for dk in delete_keys:
        del img_to_caption[dk]
    logger.info("The number of captions after deletion: %d" % len(img_to_caption.keys()))

    final_img_to_caption = {}
    for id,data in tqdm(enumerate(img_to_caption.items())):
        img_id, caption = data
        final_img_to_caption[img_id] = caption
        if max_image_num > -1:# if =-1, just use all data!
            if id >= max_image_num - 1:
                break

    logger.info("final : {}".format( len(final_img_to_caption.keys())))
    return final_img_to_caption, coco


def _build_vocab(caption_list, threshold=1):
    counter = Counter()
    max_len = 0
    for i, caption in enumerate(caption_list):
        words = caption.split(' ')  # caption contrains only lower-case words
        for w in words:
            counter[w] += 1

        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print "Max length of caption: ", max_len
    return word_to_idx


def _build_caption_vector(caption, word_to_idx, max_length=15):
    captions = np.ndarray(max_length + 2).astype(np.int32)
    words = caption.split(" ")  # caption contrains only lower-case words
    cap_vec = []
    cap_vec.append(word_to_idx['<START>'])
    for word in words:
        if word in word_to_idx: #TODO, Shouldn't  it must exist the key in word_to_idx ????
            cap_vec.append(word_to_idx[word])
    cap_vec.append(word_to_idx['<END>'])

    # pad short caption with the special null token '<NULL>' to make it fixed-size vector
    if len(cap_vec) < (max_length + 2):
        for j in range(max_length + 2 - len(cap_vec)):
            cap_vec.append(word_to_idx['<NULL>'])

    captions[:] = np.asarray(cap_vec)
    #print "Finished building caption vectors"
    return captions


def generate_mask(_coco,catId_to_ascendorder, img_id):
    img = _coco.loadImgs(img_id)[0]
    img_file_name = img['file_name']
    annIds = _coco.getAnnIds(imgIds=img_id)
    img_mask = np.zeros((img['height'], img['width'], 1), dtype=np.uint8)

    for annId in annIds:
        ann = _coco.loadAnns(annId)[0]
        img_mask, ann = generate_image_mask(_coco, img_mask, annId, catId_to_ascendorder)

    return img_file_name, img_mask



class DataLoader(RNGDataFlow):
    def __init__(self, name, max_length, img_size,
                 train_img_num = 4000, test_img_num = 1000, use_caption = True, quick_eval = True,
                 regenerate_json = True):

        self.max_length = max_length
        self.img_size = img_size
        self.name = name
        self.shuffle = False
        self.use_caption = use_caption
        self.vocab_name = "word_to_idx_max{}_train{}_test{}.json".format(max_length, train_img_num,test_img_num)
        self.quick_eval = quick_eval
        self.regenerate_json = regenerate_json

        if "train" in self.name:
            self.image_dir = coco_train_dir
            img_dict_train, coco_caption = _process_caption_data(caption_file=caption_train_json,
                                                                          max_length=self.max_length, max_image_num = train_img_num)

            if not regenerate_json:
                logger.info("load vocab from {}".format(self.vocab_name))
                with open(self.vocab_name, 'r') as load_f:
                    self.word_to_idx = json.load(load_f)
                    logger.info("vocab length: {}".format(len(self.word_to_idx.keys())))

            else:
                #generate word_to_idx vocab file both from train and test data(because in the test, we also need caption as input).
                logger.info("generating {}.....".format(self.vocab_name))
                img_dict_val, _ = _process_caption_data(caption_file=caption_val_json,
                                                                              max_length=self.max_length, max_image_num=test_img_num)
                caption_list = img_dict_train.values()
                caption_list.extend(img_dict_val.values())
                word_to_idx = _build_vocab(caption_list=caption_list, threshold=word_count_threshold)
                logger.info("build vocab done, vocab length = {}.".format(len(word_to_idx.keys())))
                self.word_to_idx = word_to_idx
                with open(self.vocab_name,"w") as f:
                    logger.info("save the vocab..")
                    json.dump(self.word_to_idx,f)

            self.shuffle = True
            self.img_ids = img_dict_train.keys()
            self.img_dict = img_dict_train
            self.coco_caption = coco_caption
            self.coco_instance = COCO(instance_train_json)



        elif "test" in self.name:
            self.image_dir = coco_val_dir
            img_dict, coco_caption = _process_caption_data(caption_file=caption_val_json,
                                                                          max_length=self.max_length, max_image_num = test_img_num)

            with open(self.vocab_name, 'r') as load_f:
                self.word_to_idx = json.load(load_f)

            self.img_ids = img_dict.keys()
            self.img_dict = img_dict
            self.coco_caption = coco_caption
            self.coco_caption = coco_caption
            self.coco_instance = COCO(instance_val_json)

        else:
            raise

        self.catId_to_ascendorder = generate_id2trainid(self.coco_instance)
        logger.info("catId_to_ascendorder length: {}".format(len(self.catId_to_ascendorder)))

        print self.catId_to_ascendorder

        logger.info("dataset size: {}".format(len(self.img_ids)))


    def size(self):
        if "train" in self.name:
            return len(self.img_dict.keys())
        elif "test" in self.name:
            if self.quick_eval:
                return 1000
            else:
                return len(self.img_dict.keys())
        else:
            raise


    @staticmethod
    def class_num():
        return class_num #Coco

    def get_data(self): # only for one-shot learning
        if self.shuffle:
            self.rng.shuffle(self.img_ids)

        for i in range(self.size()):
            if "train" in self.name:
                img_id = self.img_ids[i]
                caption = self.img_dict[img_id]# only consider one caption
                img_file_name, gt = generate_mask(self.coco_instance,self.catId_to_ascendorder, img_id)
                img = cv2.imread(os.path.join(self.image_dir,img_file_name))
                caption_ids = _build_caption_vector(caption, self.word_to_idx,max_length=self.max_length)

                img = resize_and_pad(img, self.img_size, self.img_size,interp=cv2.INTER_LINEAR)
                gt = resize_and_pad(gt, self.img_size, self.img_size, interp=cv2.INTER_NEAREST)

                if self.use_caption:
                    yield [img, gt, caption_ids]
                else:
                    yield [img, gt]
            else:
                img_id = self.img_ids[i]
                #if str(img_id) != "546229":
                #    continue
                caption = self.img_dict[img_id]  # only consider one caption
                img_file_name, gt = generate_mask(self.coco_instance, self.catId_to_ascendorder, img_id)
                img = cv2.imread(os.path.join(self.image_dir, img_file_name))

                caption_ids = _build_caption_vector(caption, self.word_to_idx, max_length=self.max_length)

                img = resize_and_pad(img, self.img_size, self.img_size, interp=cv2.INTER_LINEAR)
                gt = resize_and_pad(gt, self.img_size, self.img_size, interp=cv2.INTER_NEAREST)

                if self.use_caption:
                    yield [img, gt, caption_ids]
                else:
                    yield [img, gt]





if __name__ == '__main__':
    ds = DataLoader("test", max_length=15, img_size=320,train_img_num=-1,test_img_num=-1,regenerate_json=True)
    ds.reset_state()
    for idx, data in enumerate(ds.get_data()):
        img, gt, caption = data[0],data[1],data[2]
        print("caption str: {}".format(caption))
        #print("caption id: {}".format(data[3]))
        cv2.imshow("img", img)
        cv2.imshow("label", visualize_label(gt,class_num=class_num))
        print(np.max(gt))
        print(np.unique(gt))
        cv2.waitKey(50000)


