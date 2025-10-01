import os

class Experiment:
    
    def __init__(self, config, l1, l2):
        self.config = config
        self.list_hlp_methods = l1
        self.list_negative_samplers = l2

    def run(self):
        for _ in range(self.config['num_trials']):
            for hlp_method in self.list_hlp_methods:
                for negative_sampler in self.list_negative_samplers:
                    command = f"uv run pipeline --dataset_name {self.config['dataset_name']} --hlp_method {hlp_method} --negative_sampling {negative_sampler} --output_path {self.config['output_path']} --random_seed {self.config['random_seed']}"
                    print(f"Running command: {command}")
                    try:
                        os.system(command)
                    except Exception as e:
                        print(f"Error running command: {command}")

class Results:
    def __init__(self, path):
        self.path = path

    def to_csv(self):
        pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiments with different HLP methods and negative samplers.")
    parser.add_argument('-n', '--num_trials', type=int, default=5, help='Number of trials to run for each configuration.')
    parser.add_argument('-d', '--dataset_name', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output results.')
    parser.add_argument('-r', '--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    config = {
        'num_trials': args.num_trials,
        'dataset_name': args.dataset_name,
        'output_path': args.output_path,
        'random_seed': args.random_seed
    }

    list_hlp_methods = ['CommonNeighbors']
    list_negative_samplers = ['MotifHypergraphNegativeSampler', 'CliqueHypergraphNegativeSampler']

    experiment = Experiment(config, list_hlp_methods, list_negative_samplers)
    experiment.run()