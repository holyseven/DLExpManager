############ seg2 experiment manager ############

import json
import os.path as osp
from .exp_manager import flatten_config


def get_metric_result(logger, config):
    final_epoch = config['trainer/epochs']
    saved_last_epoch = list(logger.keys())[-1]
    
    n = int(saved_last_epoch)
    for i in range(int(saved_last_epoch), 0, -1):
        if 'eval' in logger[str(i)]:
            n = i
            break
    
    if 'eval' not in logger[str(n)]:
        return 0.0, f'0/{final_epoch}'
    else:
        return logger[str(n)]['eval']['Mean_IoU'], f'{saved_last_epoch}/{final_epoch}'

def seg2_read_one_exp(work_dir):
    train_logger_path = osp.join(work_dir, 'train_logger.json')
    val_logger_path = osp.join(work_dir, 'val_logger.json')
    config_path = osp.join(work_dir, 'config.json')
    
    if not osp.exists(config_path):
        return
    
    with open(config_path, 'r') as f:
        config = json.load(fp=f)
        flattened_config = flatten_config(config)
    
    # get metric during training
    if osp.exists(train_logger_path):
        with open(train_logger_path, 'r') as f:
            train_logger = json.load(fp=f)
            metric_during_trainining, is_finished = get_metric_result(train_logger, flattened_config)
            flattened_config['metric_training_mIoU'] = metric_during_trainining
            flattened_config['finished'] = is_finished
    else:
        flattened_config['metric_training_mIoU'] = 0.0
        flattened_config['finished'] = 'N/A'
    
    # get metric of eval.
    if osp.exists(val_logger_path):
        with open(val_logger_path, 'r') as f:
            val_logger = json.load(fp=f)
            metric, _ = get_metric_result(val_logger, flattened_config)
            flattened_config['metric_mIoU'] = metric
    else:
        flattened_config['metric_mIoU'] = 0.0
    
    return flattened_config


# import pandas as pd
# from sideman import BaseExperimentResultManager, seg2_read_one_exp
# import glob
# f_dir_pattern = '../seg3_outdir/ade*/*/*/'
# list_results = []
# for f_dir in sorted(glob.glob(f_dir_pattern)):
#     res = seg2_read_one_exp(f_dir)
#     if res is not None:
#         list_results.append(res)
    
# mag = BaseExperimentResultManager(list_results)
# df = mag.summarize()

# pd.set_option('display.max_rows', df.shape[0]+1)
# # df.sort_values('metric', ascending=False)
# df