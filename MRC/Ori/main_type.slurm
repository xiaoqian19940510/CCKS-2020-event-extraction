#!/bin/bash
#SBATCH -o output/logs2/Cla_Ori_v2_dev0.25_learningRate3e-5_trainable.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J Cla_Ori_v2_dev0.25_learningRate3e-5_trainable           # 任务名称
#SBATCH --gres=gpu:V100:2
#SBATCH -c 5
CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/Cls_data_preprocess.py
CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Cls/run_bert.py --do_data 
CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Cls/run_bert.py --do_train --save_best
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Cls/run_bert.py --do_test
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/Mrc_data_preprocess.py
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Mrc/get_data.py
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Mrc/train_start.py
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Mrc/test.py
#CUDA_VISIBLE_DEVICES=0,1 python /home/LAB/liqian/test/game/Ori/CCKS-Mrc/get_submission.py