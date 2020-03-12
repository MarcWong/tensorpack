from PIL import Image
import numpy as np
import tensorflow as tf
from math import ceil
import cv2,colorsys
import matplotlib.pyplot as plt

n_classes = 21
# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
# image mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def decode_image_label(mask):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    h, w, c = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    for j_, j in enumerate(mask[:, :, 0]):
        for k_, k in enumerate(j):
            if k < n_classes:
                outputs[j_, k_] = label_colours[k]
    img = np.array(outputs)
    return img

    
def decode_labels(mask, num_images=1):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < n_classes:
                  pixels[k_,j_] = label_colours[k] #TODO j_,k_ ?
      outputs[i] = np.array(img)
    return outputs

def prepare_label(input_batch, new_size, one_hot=True):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size H W 1].
      new_size: a tensor with new height and width.

    Returns:
      Outputs a tensor of shape [batch_size h w 21]
      with last dimension comprised of 0's and 1's only.
    """
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
          input_batch = tf.one_hot(input_batch, depth=n_classes)
    return input_batch

def inv_preprocess(imgs, num_images):
  """Inverse preprocessing of the batch of images.
     Add the mean vector and convert from BGR to RGB.
   
  Args:
    imgs: batch of input images.
    num_images: number of images to apply the inverse transformations on.
  
  Returns:
    The batch of the size num_images with the same spatial dimensions as the input.
  """
  n, h, w, c = imgs.shape
  assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
  outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
  for i in range(num_images):
    outputs[i] = (imgs[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8)
  return outputs

def visualize_label(label):
    """Color classes a good distance away from each other."""
    h, w = label.shape
    img_color = np.zeros((h, w, 3))
    for i in range(1, 151):
        img_color[label == i] = to_color(i)
    return img_color * 255  # is [0.0-1.0]  should be [0-255]


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[0]
    cols_missing = target_size[1] - img.shape[1]
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img

def predict_sliding(full_image, predictor, classes, tile_size):
    """Predict on tiles of exactly the network input shape so nothing gets squeezed."""
    overlap = 1/3
    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
    #print("Need %i x %i prediction tiles @ stride %i px" % (tile_cols, tile_rows, stride))
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = full_image[y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            # plt.imshow(padded_img)
            # plt.show()
            tile_counter += 1
            #print("Predicting tile %i" % tile_counter)
            padded_img = padded_img[None, :, :, :].astype('float32') # extend one dimension
            padded_prediction = predictor(padded_img)[0][0]
            prediction = padded_prediction[0:img.shape[0], 0:img.shape[1], :]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    # visualize normalization Weights
    # plt.imshow(np.mean(count_predictions, axis=2))
    # plt.show()
    return full_probs


def predict_multi_scale(full_image, predictor, scales, classes, sliding_evaluation):
    """Predict an image by looking at it with different scales."""
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    h_ori, w_ori = full_image.shape[:2]
    for scale in scales:
        #print("Predicting image scaled by %f" % scale)
        scaled_img = cv2.resize(full_image, scale*(h_ori, w_ori))
        if sliding_evaluation:
            scaled_probs = predict_sliding(scaled_img, predictor)
        else:
            scaled_probs = predictor(scaled_img)[0][0]
        # scale probs up to full size
        h, w = scaled_probs.shape[:2]
        probs = cv2.resize(scaled_probs, (h_ori,w_ori))
        # visualize_prediction(probs)
        # integrate probs over all scales
        full_probs += probs
    full_probs /= len(scales)
    return full_probs

def visualize_prediction(prediction):
    """Visualize prediction."""
    cm = np.argmax(prediction, axis=2) + 1
    color_cm = add_color(cm)
    plt.imshow(color_cm)
    plt.show()

def add_color(img):
    """Color classes a good distance away from each other."""
    h, w = img.shape
    img_color = np.zeros((h, w, 3))
    for i in xrange(1, 151):
        img_color[img == i] = to_color(i)
    return img_color * 255  # is [0.0-1.0]  should be [0-255]

def to_color(category):
    """Map each category color a good distance away from each other on the HSV color space."""
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v, 1, 1)