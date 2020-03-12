#!/bin/sh


python res101.slim.2branch.speedup.mcontext.240k.width512.py --gpu 5\
  --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold0_1shot_test/model-5582628\
  --test_data fold0_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.width512.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.width512.py --gpu 5\
  --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold1_1shot_test/model-5579295\
  --test_data fold1_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.width512.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.width512.py --gpu 5\
  --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold2_1shot_test/model-5575962\
  --test_data fold2_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.width512.5shot.result

python res101.slim.2branch.speedup.mcontext.240k.width512.py --gpu 5\
  --test\
  --test_load train_log/res101.slim.2branch.speedup.mcontext.240k.width512:fold3_1shot_test/model-5582628\
  --test_data fold3_5shot_test >> res101.slim.2branch.speedup.mcontext.240k.width512.5shot.result