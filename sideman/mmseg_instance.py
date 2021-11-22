############ seg2 experiment manager ############

import json
import os.path as osp
import glob
from .mmcv_config import Config
from .exp_manager import flatten_config


def find_recent_log(work_dir: str, is_eval=False) -> str:
    # return filename: str

    if is_eval:
        temp_list = glob.glob(work_dir + '/*-eval.log')
        if len(temp_list) == 0:
            return None
        
        return sorted(temp_list)[-1]
        
    config_files = glob.glob(work_dir + '/*.log')
    config_files = [f for f in config_files if '-eval.log' not in f]
    if len(config_files) == 0:
        return None
    
    return sorted(config_files)[-1]

def mmcv_read_one_exp(work_dir):
    config_files = glob.glob(work_dir + '/*.py')
    assert len(config_files) == 1
    config_file = config_files[0]
    flattened_config = flatten_config(Config._file2dict(config_file)[0])

    metric_training = get_metric_result(find_recent_log(work_dir, is_eval=False))
    for k in metric_training:
        flattened_config['metric_training_' + k] = metric_training[k]

    metric = get_metric_result(find_recent_log(work_dir, is_eval=True))
    for k in metric:
        flattened_config['metric_' + k] = metric[k]

    return flattened_config

def get_metric_result(log_filename) -> dict:
    if log_filename is None:
        print('log_filename does not exist or log_filename is None. log_filename:', log_filename)
        return {'mIoU': 0.0}
    
    lines = open(log_filename, mode='r').readlines()
    target_line = None
    for l in range(len(lines)-1, len(lines)-10, -1):
        if 'global' in lines[l]:
            target_line = lines[l]
            break
    
    if target_line is None:
        print('not found target_line')
        return {'mIoU': 0.0}
    # | global | 82.5 | 91.15 | 96.08 |
    target_miou_str = target_line.split('|')[2].strip()
    try:
        target_miou = float(target_miou_str)
    except:
        target_miou = int(target_miou_str)

    return {'mIoU': target_miou}


# from sideman import BaseExperimentResultManager, mmcv_read_one_exp
# import pandas as pd
# import glob

# f_dir_pattern = '../work_dirs_voc-a/uper*voc-a*'
# list_results = []
# for f_dir in sorted(glob.glob(f_dir_pattern)):
#     res = mmcv_read_one_exp(f_dir)
#     if res is not None:
# #         res['diflr_in_pre'] = 'diflr1' in res['name']
#         list_results.append(res)

# df = BaseExperimentResultManager(list_results).summarize(only_show_dif_args=True)
# df