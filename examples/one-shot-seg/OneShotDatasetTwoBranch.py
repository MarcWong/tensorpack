# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
import ss_datalayer


__all__ = ['OneShotDatasetTwoBranch']


class OneShotDatasetTwoBranch(RNGDataFlow):
    def __init__(self,name, image_size=(321,321)):
        settings = __import__('ss_settings')
        self.name = name
        profile = getattr(settings, name)
        profile_copy = profile.copy()
        profile_copy['deploy_mode'] = True
        profile_copy['first_shape'] = image_size
        profile_copy['second_shape'] = image_size
        dbi = ss_datalayer.DBInterface(profile)
        self.data_size = len(dbi.db_items)
        if "test" in self.name:
            self.data_size = 1000
        self.PLP = ss_datalayer.PairLoaderProcess(dbi, profile_copy)

    def size(self):
        return self.data_size

    @staticmethod
    def class_num():
        return 2

    def get_data(self): # only for one-shot learning
        for i in range(self.data_size):
            first_image_list,first_label_list,second_image, second_label, metadata = self.PLP.load_next_frame()
            yield [first_image_list,first_label_list,second_image, second_label, metadata]



if __name__ == '__main__':
    ds = OneShotDatasetTwoBranch("fold0_5shot_test")

    for idx,data in enumerate(ds.get_data()):
        metadata = data[4]
        print "{} {}   {}".format(idx, ','.join(metadata['image1_name']),metadata['image2_name'])
