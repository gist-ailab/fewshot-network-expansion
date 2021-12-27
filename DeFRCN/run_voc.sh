#!/usr/bin/env bash

# --> 2. TFA-like, i.e. run seed0~9 for robust results (G-FSOD, 80 classes)
python3 main.py --gpus 0,1,2,3 --config-file configs/voc/defrcn_gfsod_r101_novelx_10shot_seedx.yaml  \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}