import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
class Experiment:
    
    def __init__(self, config):
        self.config = config

    def run(self):
        for _ in range(self.config['num_trials']):
            for dataset_name in self.config['dataset_list']:
                for hlp_method in self.config['hlp_methods_list']:
                    for negative_sampler in self.config['negative_samplers_list']:
                        if self.config['random_seed'] is None:
                            command = f"uv run pipeline --dataset_name {dataset_name} --hlp_method {hlp_method} --negative_sampling {negative_sampler} --output_path {self.config['output_path']}"
                        else:
                            command = f"uv run pipeline --dataset_name {dataset_name} --hlp_method {hlp_method} --negative_sampling {negative_sampler} --random_seed {self.config['random_seed']} --output_path {self.config['output_path']}"
                        print(f"Running command: {command}")
                        try:
                            os.system(command)
                        except Exception as e:
                            print(f"Error running command: {command}")

class Results:
    def __init__(self, path):
        self.path = path

    def load_csv(self):
        df = pd.read_csv(self.path)
        return df
    
    def to_csv(self):
        pass

    def to_latex(self, df):
        df_latex = df        
        headers = df_latex.columns.tolist()
        new_headers = []
        for h in headers:
            if h in ['hlp_method', 'negative_sampling', 'random_seed', 'train_ratio', 'val_ratio', 'test_ratio', 'accuracy', 'precision']:
                new_headers.append(h.replace('_', '\\_'))
            else:
                new_headers.append(h)
        df_latex.columns = new_headers
        df_latex['name'] = df_latex['name'].apply(lambda x: '_'.join(x.split('_')[1:]))

        # make \rules for the latex table

        latex_table = df_latex.to_latex(index=False)
        return latex_table

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments with different HLP methods and negative samplers.")
    parser.add_argument('-n', '--num_trials', type=int, default=1, help='Number of trials to run for each configuration.')
    parser.add_argument('-d', '--datasets_list', type=str, nargs='*', default=['COURSERA'], choices=['COURSERA', 'IMDB'], help='Name of the dataset to use. Available options: COURSERA, IMDB.')
    parser.add_argument('-hlp', '--hlp_methods_list', type=str, nargs='*', default=['CommonNeighbors'], help='List of HLP methods to evaluate. Default is CommonNeighbors.')
    parser.add_argument('-neg', '--negative_samplers_list', type=str, nargs='*', default=['MotifHypergraphNegativeSampler', 'CliqueHypergraphNegativeSampler'], help='List of negative sampling methods to evaluate. Default includes MotifHypergraphNegativeSampler and CliqueHypergraphNegativeSampler.')
    parser.add_argument('-o', '--output_path', type=str, default="./results", help='Path to save the output results.')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    config = {
        'num_trials': args.num_trials,
        'dataset_list': args.datasets_list,
        'hlp_methods_list': args.hlp_methods_list,
        'negative_samplers_list': args.negative_samplers_list,
        'output_path': args.output_path,
        'random_seed': args.random_seed
    }

    print("Experiment Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    experiment = Experiment(config)
    experiment.run()

    results = Results(os.path.join(args.output_path, 'exp.csv'))
    df = results.load_csv()
    print(df)
    
    # generate a latex table from the results
    latex_table = results.to_latex(df)
    print(latex_table)