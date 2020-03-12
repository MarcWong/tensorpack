#!/usr/bin/env bash
python unet.py --data_dir \
/data_a/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012 \
--meta_dir pascalvoc12 \
--gpu 1 --crop_size 256 \
--val_crop_size 128 \
--class_num 21