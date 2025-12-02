import yaml
import os
import subprocess
import argparse
import sys

experiment_variations = {
    "drop_out": [0, 0.25, 0.75],
    "conv_filters": [32, 128],
    "layers": [2, 4],
    "batch_normalization": [False],
    "xavier_initialization": [False], 
    "kernel_sizes": [[7, 3], [3, 3]],
    "strides": [[1, 1]], 
    "dilation": [[1, 1]], 
    "activation": ["ReLu"],
    "max_pool_size": [[2, 2]], 
    "combined_dilation_strides": ["combo_1_1"] 
}

def train_exp(experiments, hyperparams, ret_days=5, model_dir='./model/sensitivity', output_dir='./output/sensitivity', seed=42, tolerence=0, patience=2):
    # --- Baseline  ---
    config = yaml.safe_load(open('config.yaml', 'r'))
    with open('temporal_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    command = [
        'python', 'train.py',
        '-model_path', model_dir,
        '-data', './monthly_20d',
        '-output', output_dir,
        '-ckpt_path', './ckpt/sensitivity',
        '-config_path', 'temporal_config.yaml',
        '-model_name', f'baseline_I20R{ret_days}',
        '-device', 'cuda',
        '-seed', f'{seed}',
        '-tolerence', f'{tolerence}',
        '-patience', f'{patience}',
        '-batch_size', '128',
        '-num_workers', '0',
        '-learning_rate', '1e-5',
        '-num_epochs', '50',
        '-ret_days', f'{ret_days}',
        '-year_start', '1993',
        '-year_split', '1999',
        '-year_end', '2019',
        '-ratio', '0.7',
        ]
    subprocess.run(command)
    if os.path.exists('temporal_config.yaml'):
        os.remove('temporal_config.yaml')

    # --- Sensitivity Experiments ---
    for i in range(len(experiments)):
        config = yaml.safe_load(open('config.yaml', 'r'))
        current_param = hyperparams[i]
        current_val = experiments[i]

        if current_param == "combined_dilation_strides":
            config['dilation'] = [1, 1]
            config['strides'] = [1, 1]
            name = "dilation_[1, 1]_strides_[1, 1]"
            
        elif current_param == "max_pool_size":
            config['pool_size'] = current_val 
            name = f"{current_param}_{current_val}"
            
        else:
            config[current_param] = current_val
            name = f"{current_param}_{current_val}"

        with open('temporal_config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
        command = [
        'python', 'train.py',
        '-model_path', model_dir,
        '-data', './monthly_20d',
        '-output', output_dir,
        '-ckpt_path', './ckpt/sensitivity',
        '-config_path', 'temporal_config.yaml',
        '-model_name', name,
        '-seed', f'{seed}',
        '-tolerence', f'{tolerence}',
        '-patience', f'{patience}',
        '-device', 'cuda',
        '-batch_size', '128',
        '-num_workers', '0',
        '-learning_rate', '1e-5',
        '-num_epochs', '50',
        '-ret_days', f'{ret_days}',
        '-year_start', '1993',
        '-year_split', '1999',
        '-year_end', '2019',
        '-ratio', '0.7',
        ]
        subprocess.run(command)
        if os.path.exists('temporal_config.yaml'):
            os.remove('temporal_config.yaml')

def eval_exp(experiments, hyperparams, ret_days=5, input_dir='./model/sensitivity', output_dir='./output/sensitivity'):
    # --- Baseline Eval ---
    config = yaml.safe_load(open('config.yaml', 'r'))
    with open('temporal_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    command = [
        'python', 'eval.py',
        '-model_path', input_dir,
        '-output', output_dir,
        '-config_path', 'temporal_config.yaml',
        '-model_name', f'baseline_I20R{ret_days}',
        '-data', './monthly_20d',
        '-device', 'cuda',
        '-batch_size', '128',
        '-num_workers', '0',
        '-ret_days', f'{ret_days}',
    ]
    subprocess.run(command)
    if os.path.exists('temporal_config.yaml'):
        os.remove('temporal_config.yaml')

    # --- Sensitivity Eval ---
    for i in range(len(experiments)):
        config = yaml.safe_load(open('config.yaml', 'r'))
        current_param = hyperparams[i]
        current_val = experiments[i]

    
        if current_param == "combined_dilation_strides":
            config['dilation'] = [1, 1]
            config['strides'] = [1, 1]
            name = "dilation_[1, 1]_strides_[1, 1]"
            
        elif current_param == "max_pool_size":
            config['pool_size'] = current_val
            name = f"{current_param}_{current_val}"
            
        else:
            config[current_param] = current_val
            name = f"{current_param}_{current_val}"
       
        with open('temporal_config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
            
        command = [
            'python', 'eval.py',
            '-model_path', input_dir,
            '-output', output_dir,
            '-config_path', 'temporal_config.yaml',
            '-model_name', name,
            '-data', './monthly_20d',
            '-device', 'cuda',
            '-batch_size', '128',
            '-num_workers', '0',
            '-ret_days', f'{ret_days}',
        ]
        subprocess.run(command)
        if os.path.exists('temporal_config.yaml'):
            os.remove('temporal_config.yaml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--ret_days', type=int, default=5, help='Number of days to predict')
    if '--train' in sys.argv:
        parser.add_argument('--model_dir', type=str, default='./model/sensitivity', help='Output directory')
        parser.add_argument('--output_dir', type=str, default='./output/sensitivity', help='Output directory')
        parser.add_argument('--tolerence', type=float, default=0, help='Tolerence for early stopping')
        parser.add_argument('--patience', type=int, default=2, help='Patience for early stopping')
        parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    if '--eval' in sys.argv:
        parser.add_argument('--output_dir', type=str, default='./output/sensitivity', help='Output directory')
        parser.add_argument('--model_dir', type=str, default='./model/sensitivity', help='Input model directory')
    args = parser.parse_args()
    
    keys, values = zip(*experiment_variations.items())
    experiments = []
    hyperparams = []
    for key, value in zip(keys, values):
        experiments.extend(value)
        hyperparams.extend([key] * len(value))
        
    if args.train:
        train_exp(experiments, hyperparams, args.ret_days, args.model_dir, args.output_dir, args.seed, args.tolerence, args.patience)
    if args.eval:
        eval_exp(experiments, hyperparams, args.ret_days, args.model_dir, args.output_dir)