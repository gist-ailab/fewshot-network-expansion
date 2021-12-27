import yaml
from ast import literal_eval as make_tuple
import subprocess

def load_yaml_file(fname):
    with open(fname, "r") as f:
        config = yaml.safe_load(f)
    return config


for split in [1]:
    for shot in [5, 10]:
        for seed in range(5):
            if seed == 0:
                seed_str = ''
            else:
                seed_str = '_seed%d' %seed

            # Base Configs
            original_config = "configs/PASCAL_VOC/%dshot_CL_IoU_merge.yml" %(split, shot)
            configs_base = load_yaml_file(original_config)
            configs_base['_BASE_'] = "Base-RCNN-FPN.yaml"
            configs_base['DATASETS']['TRAIN'] = make_tuple(configs_base['DATASETS']['TRAIN'])
            configs_base['DATASETS']['TEST'] = make_tuple(configs_base['DATASETS']['TEST'])
            
            configs_base['MODEL']['WEIGHTS'] = "/data/sung/checkpoint/few_shot/voc/split%d/base_model/model_reset_surgery.pth" %split
            configs_base['DATASETS']['TRAIN'] = ('voc_2007_trainval_all%d_%dshot%s' %(split, shot, seed_str), )
            configs_base['DATASETS']['TEST'] = ('voc_2007_test_all%d' %split, )
            configs_base['DATASET'] = 'VOC%d' %split
            configs_base['OUTPUT_DIR'] = 'checkpoints/voc%d/%dshot_seed%d_CL_IoU_base' %(split, shot, seed)
            
            # Merge Configs
            original_config = "configs/PASCAL_VOC/%dshot_CL_IoU.yml" %(split, shot)
            configs_merge = load_yaml_file(original_config)
            configs_merge['_BASE_'] = "Base-RCNN-FPN.yaml"
            configs_merge['DATASETS']['TRAIN'] = make_tuple(configs_merge['DATASETS']['TRAIN'])
            configs_merge['DATASETS']['TEST'] = make_tuple(configs_merge['DATASETS']['TEST'])
            
            configs_merge['MODEL']['WEIGHTS'] = "/data/sung/checkpoint/few_shot/voc/split%d/base_model/model_reset_surgery.pth" %split
            configs_merge['DATASETS']['TRAIN'] = ('voc_2007_trainval_all%d_%dshot%s' %(split, shot, seed_str),)
            configs_merge['DATASETS']['TEST'] = ('voc_2007_test_all%d' %split,)
            configs_merge['DATASET'] = 'VOC%d' %split
            configs_merge['OUTPUT_DIR'] = 'checkpoints/voc%d/%dshot_seed%d_CL_IoU_merge' %(split, shot, seed)
            
            # Save Configs
            with open('configs/temp_base.yml', "w") as fp:
                yaml.dump(configs_base, fp)
            
            with open('configs/temp_merge.yml', "w") as fp:
                yaml.dump(configs_merge, fp)


            cmd = "python3 -m tools.train_net --gpus 0,1,2,3,4,5,6,7 --config-file configs/temp_base.yml --exp_name fsce_merge_voc%d_%dshot_seed%d" %(split, shot, seed)
            subprocess.call(cmd, shell=True)

            cmd = "python3 -m tools.train_net --gpus 0,1,2,3,4,5,6,7 --config-file configs/temp_merge.yml --exp_name fsce_base_voc%d_%dshot_seed%d" %(split, shot, seed)
            subprocess.call(cmd, shell=True)
