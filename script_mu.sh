#!/bin/bash

# mu accelerated L=30
/home/anvuong/miniconda3/envs/tf14/bin/python run.py --datadir /mnt/i/dataset-processed --algorithm mu --accelerated --L 30 --alpha 2 --eps 0.1 --max-iter $1 --outputdir ./MU_Accel_L30 --use-sample

# mu accelerated L=60
/home/anvuong/miniconda3/envs/tf14/bin/python run.py --datadir /mnt/i/dataset-processed --algorithm mu --accelerated --L 60 --alpha 2 --eps 0.1 --max-iter $1 --outputdir ./MU_Accel_L60 --use-sample

# mu L=30
/home/anvuong/miniconda3/envs/tf14/bin/python run.py --datadir /mnt/i/dataset-processed --algorithm mu --L 30 --max-iter $1 --outputdir ./MU_L30 --use-sample

# mu L=60
/home/anvuong/miniconda3/envs/tf14/bin/python run.py --datadir /mnt/i/dataset-processed --algorithm mu --L 60 --max-iter $1 --outputdir ./MU_L60 --use-sample