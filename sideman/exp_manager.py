import pandas as pd

class BaseExperimentResultManager(object):
    """Typically this is a 2D database.

    """
    def __init__(self, list_results: list) -> None:
        """[summary]

        Args:
            list_results (list): a list of dict, with keys being the hyper-parameter names,
                and values being the hyper-parameters.
        """
        super().__init__()

        self.list_results = list_results  # a list of dict.
        
        self.check_list_results()

    def check_list_results(self):
        assert isinstance(self.list_results, list)
        assert isinstance(self.list_results[0], dict)

        all_args = list(self.list_results[0].keys())
        for r in self.list_results:
            assert list(r.keys()) == all_args

    def filter_results(self, **args):
        # e.g., remove unfinished experiments.
        # TODO

        # e.g., no filters
        return self.list_results

    def hide_common_args(self, metric_name='metric'):
        """[summary]

        Args:
            metric_name (str, optional): metric_name is always shown. It is a pattern str that 
            all args contain this str will be shown. Defaults to 'metric'.

        Returns:
            [list]: a list of dict.
        """
        if len(self.list_results) == 1:
            return self.list_results

        common_args = []
        dif_args = []
        all_args = list(self.list_results[0].keys())
        for arg in all_args:
            common = True
            for r in self.list_results:
                if r[arg] != self.list_results[0][arg]:
                    common = False
                    break
            if common and (metric_name not in arg):
                common_args.append(arg)
            else:
                dif_args.append(arg)
        
        reduced_results = []  # results without common args.
        for r in self.list_results:
            new_r = {arg: r[arg] for arg in dif_args}
            reduced_results.append(new_r)

        return reduced_results

    def summarize(self, metric_name='metric', only_show_dif_args: bool=True, hiding_args=None):
        # find common args and hide them if only_show_dif_args.
        results_to_print = self.hide_common_args(metric_name) if only_show_dif_args else self.list_results
        
        # other args to hide. some args may contain less useful info so we may not want to show them.
        if hiding_args is not None:
            if isinstance(hiding_args, str):
                hiding_args = [hiding_args]
            assert isinstance(hiding_args, list)
            
            temp_list = []
            for r in results_to_print:
                for k in hiding_args:
                    del r[k]
                temp_list.append(r)
            results_to_print = temp_list

        # Use pandas.DataFrame to show the results.
        df = pd.DataFrame(columns=list(results_to_print[0].keys()))
        for i, r in enumerate(results_to_print):
            df.loc[i] = list(r.values())

        return df


############ seg2 experiment manager ############

import json
import os.path as osp

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

def flatten_config(config):
    flattened_config = {}
    for k in config.keys():
        if isinstance(config[k], dict):
            t_flattened_config = flatten_config(config[k])
            for t_k in t_flattened_config.keys():
                flattened_config[k + '/' + t_k] = t_flattened_config[t_k]
        else:
            flattened_config[k] = config[k]
    
    return flattened_config

def read_one_exp(work_dir):
    train_logger_path = osp.join(work_dir, 'train_logger.json')
    val_logger_path = osp.join(work_dir, 'val_logger.json')
    config_path = osp.join(work_dir, 'config.json')
    
    if not osp.exists(config_path):
        return
    
    with open(config_path, 'r') as f:
        config = json.load(fp=f)
        flattened_config = flatten_config(config)
    
    if osp.exists(train_logger_path):
        with open(train_logger_path, 'r') as f:
            train_logger = json.load(fp=f)
            metric_during_trainining, is_finished = get_metric_result(train_logger, flattened_config)
            flattened_config['metric_training'] = metric_during_trainining
            flattened_config['finished'] = is_finished
    else:
        flattened_config['metric_training'] = 0.0
        flattened_config['finished'] = 'N/A'
    
    if osp.exists(val_logger_path):
        with open(val_logger_path, 'r') as f:
            val_logger = json.load(fp=f)
            metric, _ = get_metric_result(val_logger, flattened_config)
            flattened_config['metric'] = metric
    else:
        flattened_config['metric'] = 0.0
    
    return flattened_config


def camvid_results():
    import glob
    f_dir_pattern = '../seg3_outdir/cam*/*/*/'
    list_results = []
    for f_dir in sorted(glob.glob(f_dir_pattern)):
        res = read_one_exp(f_dir)
        if res is not None:
            list_results.append(res)
        
    mag = BaseExperimentResultManager(list_results)
    df = mag.summarize()

    pd.set_option('display.max_rows', df.shape[0]+1)
    df.sort_values('metric', ascending=False)
