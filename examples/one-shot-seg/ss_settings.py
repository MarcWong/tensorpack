import numpy as np
import os.path as osp
from util import Map



# Classes in pascal dataset
PASCAL_CATS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
               'train', 'tv/monitor']

# Download Pascal VOC from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
PASCAL_PATH= '/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
SBD_PATH = '/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012'
image_size = (321, 321)


def get_cats(split, fold, num_folds=4):
    '''
      Returns a list of categories (for training/test) for a given fold number

      Inputs:
        split: specify train/val
        fold : fold number, out of num_folds
        num_folds: Split the set of image classes to how many folds. In BMVC paper, we use 4 folds

    '''
    num_cats = len(PASCAL_CATS)
    assert(num_cats%num_folds==0)
    val_size = int(num_cats/num_folds)
    assert(fold<num_folds)
    val_set = [ fold*val_size+v for v in range(val_size)]
    train_set = [x for x in range(num_cats) if x not in val_set]
    if split=='train':
        return [PASCAL_CATS[x] for x in train_set] 
    else:
        return [PASCAL_CATS[x] for x in val_set] 


########################### The default setting ##########################################

empty_profile = Map(
                ###############################################
                # Do not change this part
                first_label_params=[('first_label', 1., 0.)],
                second_label_params=[('second_label', 1., 0.)],
                ###############################################
                k_shot=1,
                first_shape=None,
                second_shape=None,
                output_type=None,
                read_mode=None, # Either "Shuffle" (for training) or "Deterministic" (for testing, random seed fixed)
                scale_256=True,
                mean = (104.0/255, 116.0/255, 122.0/255),
                batch_size = 1,
                video_sets=[],
                image_sets=[],
                areaRng = [0 , np.inf],
                default_pascal_cats = PASCAL_CATS,
                default_coco_cats = None,
                pascal_cats = PASCAL_CATS,
                coco_cats = None,
                coco_path = None,
                pascal_path = PASCAL_PATH,
                sbd_path = SBD_PATH,
                worker_num = 4)


########################### Settings for reproducing experiments ###########################


foldall_train = Map(empty_profile,
    read_mode='shuffle',
    image_sets=['sbd_training', 'pascal_training'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size) # original code is second_shape=None),TODO

foldall_1shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['pascal_test'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size,
     k_shot=1 ) # original code is second_shape=None),TODO

foldall_5shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['pascal_test'],
    #image_sets=['pascal_training'],
    pascal_cats = PASCAL_CATS,
    first_shape=image_size,
    second_shape=image_size,
     k_shot=5 ) # original code is second_shape=None),TODO


#### fold 0 ####

# Setting for training (on **training images**)
fold0_train = Map(empty_profile,
    read_mode='shuffle',
    image_sets=['sbd_training', 'pascal_training'],
    #image_sets=['pascal_training'],
    pascal_cats = get_cats('train',0),
    first_shape=image_size,
    second_shape=image_size) # original code is second_shape=None),TODO

fold0_5shot_train = Map(fold0_train,k_shot=5)

# Setting for testing on **test images** in unseen image classes (in total 5 classes), 5-shot
fold0_5shot_test = Map(empty_profile,
    db_cycle = 1000,
    read_mode='deterministic',
    image_sets=['pascal_test'],
    pascal_cats = get_cats('test',0),
    first_shape=image_size,
    second_shape=image_size,
    k_shot=5)

## Setting for testing on **test images** in unseen image classes (in total 5 classes), 1-shot
fold0_1shot_test = Map(fold0_5shot_test, k_shot=1)
fold0_2shot_test = Map(fold0_5shot_test, k_shot=2)
fold0_3shot_test = Map(fold0_5shot_test, k_shot=3)
fold0_4shot_test = Map(fold0_5shot_test, k_shot=4)
fold0_5shot_test = Map(fold0_5shot_test, k_shot=5)
fold0_6shot_test = Map(fold0_5shot_test, k_shot=6)
fold0_7shot_test = Map(fold0_5shot_test, k_shot=7)
fold0_8shot_test = Map(fold0_5shot_test, k_shot=8)
fold0_9shot_test = Map(fold0_5shot_test, k_shot=9)
fold0_10shot_test = Map(fold0_5shot_test, k_shot=10)


#### fold 1 ####
fold1_train = Map(fold0_train, pascal_cats=get_cats('train', 1))
fold1_5shot_train = Map(fold1_train,k_shot=5)
fold1_5shot_test = Map(fold0_5shot_test, pascal_cats=get_cats('test', 1))
fold1_1shot_test = Map(fold1_5shot_test, k_shot=1)

#### fold 2 ####
fold2_train = Map(fold0_train, pascal_cats=get_cats('train', 2))
fold2_5shot_train = Map(fold2_train,k_shot=5)
fold2_5shot_test = Map(fold0_5shot_test, pascal_cats=get_cats('test', 2))
fold2_1shot_test = Map(fold2_5shot_test, k_shot=1)

#### fold 3 ####
fold3_train = Map(fold0_train, pascal_cats=get_cats('train', 3))
fold3_5shot_train = Map(fold3_train,k_shot=5)
fold3_5shot_test = Map(fold0_5shot_test, pascal_cats=get_cats('test', 3))
fold3_1shot_test = Map(fold3_5shot_test, k_shot=1)



