#!/bin/sh


python res101.slim.2branch.speedup.240k.forbash.py --gpu 3\
  --train_data fold0_train\
  --test_data fold0_1shot_test >> result_txt

python res101.slim.2branch.speedup.240k.forbash.py --gpu 3\
  --train_data fold1_train\
  --test_data fold1_1shot_test >> result_txt

python res101.slim.2branch.speedup.240k.forbash.py --gpu 3\
  --train_data fold2_train\
  --test_data fold2_1shot_test >> result_txt

python res101.slim.2branch.speedup.240k.forbash.py --gpu 3\
  --train_data fold3_train\
  --test_data fold3_1shot_test >> result_txt