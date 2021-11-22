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
        print('<summarize> found these args in the original results:', list(self.list_results[0].keys()))
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
        print('<summarize> only these args to print:', list(results_to_print[0].keys()))
        df = pd.DataFrame(columns=list(results_to_print[0].keys()), dtype=object)
        for i, r in enumerate(results_to_print):
            df.loc[i] = list(r.values())

        return df


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