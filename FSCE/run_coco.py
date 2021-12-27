import yaml
from ast import literal_eval as make_tuple
import subprocess

def load_yaml_file(fname):
    with open(fname, "r") as f:
        config = yaml.safe_load(f)
    return config


for shot in [10, 30]:
    for seed in range(5):
        if seed == 0:
            seed_str = ''
        else:
            seed_str = '_seed%d' %seed

        # Base Configs
        original_config = "configs/COCO/%dshot_baseline_merge.yml" %shot
        configs_base = load_yaml_file(original_config)
        
        configs_base['_BASE_'] = "Base-RCNN-FPN.yaml"
        configs_base['DATASETS']['TRAIN'] = make_tuple(configs_base['DATASETS']['TRAIN'])
        configs_base['DATASETS']['TEST'] = make_tuple(configs_base['DATASETS']['TEST'])
        
        configs_base['MODEL']['WEIGHTS'] = "/data/sung/checkpoint/few_shot/coco/base_model/model_reset_surgery.pth"
        configs_base['DATASETS']['TRAIN'] = ('coco_trainval_all_%dshot%s' %(shot, seed_str), )
        configs_base['DATASETS']['TEST'] = ('coco_test_all', )
        configs_base['DATASET'] = 'COCO'
        configs_base['OUTPUT_DIR'] = 'checkpoints/COCO/%dshot_seed%d_base' %(shot, seed)
        
        # Merge Configs
        original_config = "configs/COCO/%dshot_baseline.yml" %shot
        configs_merge = load_yaml_file(original_config)
        
        configs_merge['_BASE_'] = "Base-RCNN-FPN.yaml"
        configs_merge['DATASETS']['TRAIN'] = make_tuple(configs_merge['DATASETS']['TRAIN'])
        configs_merge['DATASETS']['TEST'] = make_tuple(configs_merge['DATASETS']['TEST'])
        
        configs_merge['MODEL']['WEIGHTS'] = "/data/sung/checkpoint/few_shot/coco/base_model/model_reset_surgery.pth"
        configs_merge['DATASETS']['TRAIN'] = ('coco_trainval_all_%dshot%s' %(shot, seed_str), )
        configs_merge['DATASETS']['TEST'] = ('coco_test_all', )
        configs_merge['DATASET'] = 'COCO'
        configs_merge['OUTPUT_DIR'] = 'checkpoints/COCO/%dshot_seed%d_merge' %(shot, seed)

        
        # Save Configs
        with open('configs/temp_base.yml', "w") as fp:
            yaml.dump(configs_base, fp)
        
        with open('configs/temp_merge.yml', "w") as fp:
            yaml.dump(configs_merge, fp)


        cmd = "python3 -m tools.train_net --gpus 0,1,2,3,4,5,6,7 --config-file configs/temp_base.yml --exp_name fsce_merge_coco_%dshot_seed%d" %(shot, seed)
        subprocess.call(cmd, shell=True)

        cmd = "python3 -m tools.train_net --gpus 0,1,2,3,4,5,6,7 --config-file configs/temp_merge.yml --exp_name fsce_base_coco_%dshot_seed%d" %(shot, seed)
        subprocess.call(cmd, shell=True)
