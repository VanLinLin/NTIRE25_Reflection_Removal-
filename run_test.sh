#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29508 ./test_reflection.py --win_size 8 --save_images