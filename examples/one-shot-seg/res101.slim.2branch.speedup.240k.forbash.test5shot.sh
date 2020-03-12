#!/bin/sh


python res101.slim.2branch.speedup.240k.forbash.py --gpu 1\
  --test\
  --test_data fold0_5shot_test\
  --test_load train_log/res101.slim.2branch.speedup.240k.forbash:fold0_1shot_test/model-5582628\
   >> result_test_txt_5shot

python res101.slim.2branch.speedup.240k.forbash.py --gpu 1\
  --test\
  --test_data fold1_5shot_test\
   --test_load train_log/res101.slim.2branch.speedup.240k.forbash:fold1_1shot_test/model-5582628\
    >> result_test_txt_5shot

python res101.slim.2branch.speedup.240k.forbash.py --gpu 1\
  --test\
  --test_data fold2_5shot_test\
   --test_load train_log/res101.slim.2branch.speedup.240k.forbash:fold2_1shot_test/model-5579295\
    >> result_test_txt_5shot

python res101.slim.2branch.speedup.240k.forbash.py --gpu 1\
  --test\
  --test_data fold3_5shot_test\
   --test_load train_log/res101.slim.2branch.speedup.240k.forbash:fold3_1shot_test/model-5582628\
    >> result_test_txt_5shot