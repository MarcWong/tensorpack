import cv2
import numpy as np


pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]]) # BGR

def draw_skeleton(input_image, sk):
    for j in range(sk.shape[0]):
        cv2.circle(input_image, tuple(sk[j]), 2, tuple((255, 0, 0)), 2)
    cv2.line(input_image, tuple(sk[0]), tuple(sk[1]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[1]), tuple(sk[2]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[2]), tuple(sk[6]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[6]), tuple(sk[3]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[3]), tuple(sk[4]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[4]), tuple(sk[5]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[6]), tuple(sk[7]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[7]), tuple(sk[8]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[9]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[12]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[12]), tuple(sk[11]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[11]), tuple(sk[10]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[8]), tuple(sk[13]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[13]), tuple(sk[14]), (0, 255, 255), 2)
    cv2.line(input_image, tuple(sk[14]), tuple(sk[15]), (0, 255, 255), 2)




def visualize(oriImg, points, pa):
    import matplotlib
    import cv2 as cv
    import matplotlib.pyplot as plt
    import math

    fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170,0,255],[255,0,255]]
    canvas = oriImg
    stickwidth = 4
    x = points[:, 0]
    y = points[:, 1]

    for n in range(len(x)):
        for child in range(len(pa)):
            if pa[child] is 0:
                continue

            x1 = x[pa[child] - 1]
            y1 = y[pa[child] - 1]
            x2 = x[child]
            y2 = y[child]

            cv.line(canvas, (x1, y1), (x2, y2), colors[child], 8)


    plt.imshow(canvas[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)

    from time import gmtime, strftime
    import os
    directory = 'data/mpii/result/test_images'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, strftime("%Y-%m-%d-%H_%M_%S", gmtime()) + '.jpg')

    plt.savefig(fn)


def crop_and_padding(img_path, objcenter, scale, joints, data_shape, output_shape, stage):
    nr_skeleton = 16
    img = cv2.imread(img_path)
    add = max(img.shape[0], img.shape[1])
    big_img = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                 value=pixel_means.reshape(-1))

    zero_index = np.where(joints[:, 0] == 0)
    joints[:, 0] += add
    joints[:, 1] += add
    objcenter[0] += add
    objcenter[1] += add
    ###################################################### here is one cheat
    if stage == 'train':
        ext_ratio = 1.25
    elif stage == 'valid':
        ext_ratio = 1.25
    else:
        ext_ratio = 1.

    delta = int(scale * ext_ratio) // 2
    min_x = int(objcenter[0] - delta)
    max_x = int(objcenter[0] + delta)
    min_y = int(objcenter[1] - delta)
    max_y = int(objcenter[1] + delta)

    joints[:, 0] = joints[:, 0] - min_x  # (0,0,1) means empty point, just set it zero, otherwise will be negtive
    joints[:, 1] = joints[:, 1] - min_y

    x_ratio = float(output_shape[0]) / (max_x - min_x)
    y_ratio = float(output_shape[1]) / (max_y - min_y)

    joints[:, 0] *= x_ratio
    joints[:, 1] *= y_ratio

    img = cv2.resize(big_img[min_y:max_y, min_x:max_x, :], (data_shape[0], data_shape[1]))

    joints[zero_index] = np.array([0, 0, 0])  # set out-border keypoint to left-top postion ground truth

    label = joints[:, :2]

    if False:
        from tensorpack.utils.skeleton.visualization import draw_skeleton
        img = draw_skeleton(img, 4 * label.astype(int))
        print(label)
        print("scale: {}".format(scale))
        cv2.imshow('', img)
        cv2.waitKey(4000)

    final_label = np.zeros((nr_skeleton, output_shape[0], output_shape[1]))
    for i in range(nr_skeleton):
        # if tpts[i, 2] > 0: # This is evil!!
        if label[i, 0] < output_shape[0] and label[i, 1] < output_shape[1] \
                and label[i, 0] > 0 and label[i, 1] > 0:
            final_label[i, int(label[i, 1]), int(label[i, 0])] = 1  # here, notice the order of opencv

    final_label = np.transpose(final_label, (1, 2, 0))
    final_label = cv2.GaussianBlur(final_label, (7, 7), 0)

    for i in range(
            nr_skeleton):  # normalize to 1, otherwise the peak value may be 0.25, please notice the cv2.GaussianBlur's result.
        am = np.amax(final_label[:, :, i])
        if am == 0:
            continue
        final_label[:, :, i] /= am / 255  # normalize to 1

    transform = {}
    transform['divide_first'] = (x_ratio, y_ratio)
    transform['add_second'] = (min_x - add, min_y - add)

    return img, final_label, transform



def preprocess(current_skeleton_obj,data_shape,output_shape,stage):
    annolist_index = current_skeleton_obj["annolist_index"]
    img_paths = current_skeleton_obj["img_paths"]
    height = current_skeleton_obj['img_height']
    width = current_skeleton_obj['img_width']
    joint_self = current_skeleton_obj["joint_self"]
    # TODO Adjust center/scale slightly to avoid cropping limbs
    # For single-person pose estimation with a centered/scaled figure
    center = current_skeleton_obj['objpos']
    scale = current_skeleton_obj['scale_provided'] * 200
    joint_self = np.array(joint_self)

    image, heatmap, transform_dict = crop_and_padding(img_path=img_paths,
                                                           objcenter=center, scale=scale, joints=joint_self,
                                                           data_shape=data_shape, output_shape=output_shape,
                                                           stage=stage)

    # Meta info
    meta = {'transform': transform_dict, "meta": current_skeleton_obj}

    return image, heatmap, meta


def add_flip(mypredictor, image,predict_heatmap):
    fliped = cv2.flip(image, 1)
    fliped_heatmap = mypredictor(fliped)#H,W,C
    symmetry = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)] # for mpii
    #symmetry = [ (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) ] #for coco
    for (q, w) in symmetry:# notice
        fliped_heatmap[:,:,q], fliped_heatmap[:,:,w] = fliped_heatmap[:,:,w], fliped_heatmap[:,:,q]
    return cv2.flip(fliped_heatmap,1) + predict_heatmap


def add_multiscale(mypredictor, image, predict_heatmap,scale):
    w,h,c = image.shape
    for sc in scale:
        resized_image = cv2.resize(image,(int(w*sc),int(h*sc)))
        resized_heatmap = mypredictor(resized_image)
        predict_heatmap += cv2.resize(resized_heatmap,(w,h))
    return predict_heatmap
