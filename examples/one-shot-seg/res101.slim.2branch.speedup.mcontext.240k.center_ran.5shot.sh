#!/bin/sh


python res101.slim.2branch.speedup.mcontext.240k.center_ran.py --gpu 2\
 --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran:fold0_1shot_test/model-5575962  \
  --test_data fold0_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.center_ran.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.center_ran.py --gpu 2\
 --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran:fold1_1shot_test/model-5575962  \
  --test_data fold1_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.center_ran.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.center_ran.py --gpu 2\
 --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran:fold2_1shot_test/model-5575962  \
  --test_data fold2_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.center_ran.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.center_ran.py --gpu 2\
 --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.center_ran:fold3_1shot_test/model-5582628  \
  --test_data fold3_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.center_ran.5shot.result