#!/bin/sh


python res101.slim.2branch.speedup.mcontext.240k.support_ran.py --gpu 1\
  --train_data fold0_train\
  --test_data fold0_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.support_ran.batch.result

python res101.slim.2branch.speedup.mcontext.240k.support_ran.py --gpu 1\
  --train_data fold1_train\
  --test_data fold1_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.support_ran.batch.result


python res101.slim.2branch.speedup.mcontext.240k.support_ran.py --gpu 1\
  --train_data fold2_train\
  --test_data fold2_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.support_ran.batch.result


python res101.slim.2branch.speedup.mcontext.240k.support_ran.py --gpu 1\
  --train_data fold3_train\
  --test_data fold3_1shot_test >> res101.slim.2branch.speedup.mcontext.240k.support_ran.batch.result
