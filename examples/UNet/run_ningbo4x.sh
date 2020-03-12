#!/usr/bin/env bash
python unet.ningbo4x.py --data_dir /data_a/dataset/ningbo4763 \
 --meta_dir ningbo4763 --gpu 3 \
--crop_size 384 \
--val_crop_size 384 \
--batch_size 18 \
--view