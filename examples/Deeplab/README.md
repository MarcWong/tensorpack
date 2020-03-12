
## DeepLab

more experimental results on pascalVOC can be seen in [NOTE.md](NOTE.md)

new note: [NOTE2.md](NOTE2.md)

more experimental results on RemoteSense Images can be seen in [aerial/README.md](aerial/README.md)

one shot seg can be seen in [README.md](https://github.com/dongzhuoyao/tensorpack/blob/deeplab/examples/one-shot-seg/README.md)

### Result On Pasal VOC

MSF on pascalvoc MSF all tild size: 321*321

MSC+Coco+Aug+ASPP+CRF=77.69% mIoU

Arch | Val mIoU
------------ | -------------
deeplabv2.voc.imagesize473| **71.5%** 


if without imgaug.RandomResize, the mIoU will be 68.4% rather than 69.3%.

### Result On Cityscapes

Full+Aug+ASPP+CRF=71.4% mIoU

Arch  |Val mIoU
------------ | -------- 
deeplabv2.cs.imagesize672.scale18.py| **74.21%**

## BN strategy

* standard gamma,beta are updated in 4 gpus. please note that mean,variance doesn't involve in back propagation.
* EMA of the mean, variance is maintained in the main_training_tower. 
* when back propagation, the gamma, beta are averaged over 4gpus just like other variable.
* when inference, we use the EMA of mean, variance in main_training_tower, and use the averaged gamma,beta.
* when fine tuning to segmentation task, all gamma, beta are trained,(in this case you must make the batch size as large as possible to gain a stable BN statistics). the moving_mean, moving_var is not restored from model file.


## PascalVOC 2012 preparation

 We have preprocessed the augmented images for you, you can download here, unzip it and move it into  VOCdevkit/VOC2012
 
 [https://www.dropbox.com/s/ylukfny2j1g55dq/SegmentationClassAug.zip?dl=0](https://www.dropbox.com/s/ylukfny2j1g55dq/SegmentationClassAug.zip?dl=0)
 
 
 PascalVOC 2012: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
 
 SBD: [http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
 
 
 
## TODO

* DeformableConvolution [https://github.com/Zardinality/TF-deformable-conv](https://github.com/Zardinality/TF-deformable-conv)
* WideResNet
* ResNext on cifar10(which in details: ResNeXt-29, 8×64d with baseline wideResNet)
* SE-ResNeXt-101+Mixup(currently best basic model) on cifar10
* GCN
* FRRN
* DCN
* PointNet

## Note

* Deeplabv3: The proposed model is
trained with output stride = 16, and then during inference
we apply output stride = 8 to get more detailed feature
map.interestingly, when evaluating our best cascaded model with output stride = 8, the performance
improves over evaluating with output stride = 16 by 1:39%.
