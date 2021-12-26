python3 -m tools.train_net --gpus 4,5,6,7 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU_merge.yml \
        --exp_name fsce_merge

python3 -m tools.train_net --gpus 4,5,6,7 \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --exp_name fsce_base