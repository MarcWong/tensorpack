from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
from utils import visualize
import numpy as np
import os
from tensorpack.utils import logger
from termcolor import colored
from tabulate import tabulate

"""
detection = loadmat('evaluation/detections.mat')
det_idxs = detection['RELEASE_img_index']
debug = 0
"""

def pckh(preds):
    threshold = 0.5
    SC_BIAS = 0.6



    dict = loadmat('evaluation/detections_our_format.mat')
    dataset_joints = dict['dataset_joints'] #shape
    jnt_missing = dict['jnt_missing'] #shape
    pos_gt_src = dict['pos_gt_src'] #shape

    headboxes_src = dict['headboxes_src'] #shape

    '''
    debug_length = 200
    pos_gt_src = pos_gt_src[:,:,:debug_length]
    jnt_missing =jnt_missing[:,:debug_length]
    headboxes_src = headboxes_src[:,:,:debug_length]
    '''


    pos_pred_src = transpose(preds, [1, 2, 0]) # shape 2,14, 2970?

    """
    if debug:
        for i in range(len(det_idxs[0])):
            anno = mat['RELEASE']['annolist'][0, 0][0][det_idxs[0][i] - 1]
            fn = anno['image']['name'][0, 0][0]
            imagePath = 'data/mpii/images/' + fn
            oriImg = sio.imread(imagePath)
            pred = pos_pred_src[:, :, i]
            visualize(oriImg, pred, pa)
    """

    head = np.where(dataset_joints == 'head')[1][0]
    lsho = np.where(dataset_joints == 'lsho')[1][0]
    lelb = np.where(dataset_joints == 'lelb')[1][0]
    lwri = np.where(dataset_joints == 'lwri')[1][0]
    lhip = np.where(dataset_joints == 'lhip')[1][0]
    lkne = np.where(dataset_joints == 'lkne')[1][0]
    lank = np.where(dataset_joints == 'lank')[1][0]

    rsho = np.where(dataset_joints == 'rsho')[1][0]
    relb = np.where(dataset_joints == 'relb')[1][0]
    rwri = np.where(dataset_joints == 'rwri')[1][0]
    rkne = np.where(dataset_joints == 'rkne')[1][0]
    rank = np.where(dataset_joints == 'rank')[1][0]
    rhip = np.where(dataset_joints == 'rhip')[1][0]

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src # key setence
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale) # divide an uniform scale
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible) # only consider visible points
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)


    # save
    rng = np.arange(0, 0.5, 0.01)
    pckAll = np.zeros((len(rng), 16))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
        pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True
    data = [["Hourglass", PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
            , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]),\
             0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
            , 0.5 * (PCKh[lank] + PCKh[rank]), "{}".format(np.mean(PCKh))]]
    table = tabulate(data, headers=["Model", "Head", "Shoulder", "Elbow", "Wrist", "Hip" , "Knee", "Ankle", "Mean"])
    logger.info(colored("Validation result:", 'cyan'))
    logger.info(colored("\n"+table, 'cyan'))

