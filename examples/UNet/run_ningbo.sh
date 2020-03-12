#!/usr/bin/env bash
python unet.py --data_dir /data_a/dataset/ningbo3539 \
 --meta_dir ningbo --gpu 2 \
--crop_size 256 \
--val_crop_size 256 \
--load train_log/unet_backup2/model-45280