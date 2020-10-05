#!/bin/bash
#SBATCH -o output/logs2/Fin__V3.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J Fin__V3           # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/Cls_data_preprocess.py
CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Cls/run_bert.py --do_data 
CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Cls/run_bert.py --do_train --save_best
CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Cls/run_bert.py --do_test
# CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/Mrc_data_preprocess.py
# CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Mrc/get_data.py
# CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Mrc/train_start.py
# CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Mrc/test.py
# CUDA_VISIBLE_DEVICES=0 python /home/LAB/liqian/test/game/Fin/CCKS-Mrc/get_submission.py