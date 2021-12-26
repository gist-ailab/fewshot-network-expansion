python3 -m tools.train_net --gpus 0,1,2,3 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml \
        --exp_name fsdet_base

python3 -m tools.train_net --num-gpus 0,1,2,3 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot_merge.yaml \
        --exp_name fsdet_merge
