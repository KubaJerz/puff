from expt_runner import Expt_Runner
import tomllib
import os
import random 
import itertools
import toml

class ExperimentBuilder():
    def __init__(self, toml_file_path):
        self.toml_file_path = toml_file_path
        self.sub_runs_list = []
        self.run_on_gpu = False
        self.meta_data = {}
        self.static = {}
        self.sweep = {}
        self.expt_dir = None

        self._parse_toml()

    def build_experiment_runs(self):
        if self._is_hyperparameter_sweep():
            if self.sweep['sampling_strategy'].lower() == 'random_search': 
                self.sub_runs_list = [self._random_sample() for i in range(self.sweep['num_runs'])]
            else:
                self.sub_runs_list = self._grid_sample()
        else:
            params = self._setup_static_run()
            self.sub_runs_list = [params.copy() for i in range(self.static['num_runs'])]
        
    def start_runs(self):
        if not self.sub_runs_list:
            print("ERROR: 'sub_runs_list' is empty you must build experiment first to populate array")
            return
        
        runner = Expt_Runner(expt_dir=self.expt_dir, sub_runs_list= self.sub_runs_list, run_on_gpu=self.run_on_gpu)
        runner.run()

    def _parse_toml(self):
        try:
            if not os.path.exists(self.toml_file_path):
                raise FileNotFoundError(f"Config file not found: {self.toml_file_path}")
            with open(self.toml_file_path, 'rb') as f:
                config = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML syntax in {self.toml_file_path}: {e}")


        self.meta_data = config['meta_data']       
        base_dir = self.meta_data['expt_dir']
        expt_name = self.meta_data['name']

        #make experiment dir 
        self.expt_dir = os.path.join(base_dir, f"{expt_name}")
        os.mkdir(self.expt_dir)

        #save expt toml to expt path
        expt_config_path = os.path.join(self.expt_dir, f"{expt_name}.toml")
        try:
            with open(expt_config_path, 'w') as f:
                toml.dump(config,f)
        except Exception as e:
            raise RuntimeError(f"Not able to write to {expt_config_path}: {e}")

        
        if self._is_hyperparameter_sweep():
            self.sweep = config['sweep']
            self.run_on_gpu = self.sweep['run_on_gpu']
        else:
            self.static = config['static']
            self.run_on_gpu = self.static['run_on_gpu']
    
    def _is_hyperparameter_sweep(self):
        return (self.meta_data['run_type']).lower() == "sweep"
    
    def _setup_static_run(self):
        train_run_config = {
            'epochs': self.static['epochs'] ,
            'patience': self.static.get('patience', 10), 
            'plot_freq': self.meta_data['plot_freq'],
            'model': self._build_model_config(self.static['model'] ),
            'optimizer': self._build_optimizer_config(self.static['optimizer'] ),
            'criterion': self._build_criterion_config(self.static['criterion'] ),
            'data': self._build_data_config(self.static['data'])
        }
        return train_run_config

    def _random_sample(self):
        search_space = self.sweep['search_space']
        
        train_run_config = {
            'epochs': self.sweep['epochs'] ,
            'patience': self.sweep.get('patience', 10),
            'plot_freq': self.meta_data['plot_freq'],
            'model': self._build_model_config(self.sweep['model'] ),
            'optimizer': self._build_optimizer_config(self.sweep['optimizer'] ),
            'criterion': self._build_criterion_config(self.sweep['criterion'] ),
            'data': self._build_data_config(self.sweep['data'])
        }
        
        sampled_params = {}
        for param, values in search_space.items():
            if param == 'ranges':
                continue  
            if isinstance(values, list):
                sampled_params[param] = random.choice(values)
        
        self._apply_sampled_params(train_run_config, sampled_params)
        
        return train_run_config
    
    def _grid_sample(self):
        search_space = self.sweep.get('search_space', {})
        
        param_names = []
        param_values = []
        for param, values in search_space.items():
            if param == 'ranges':
                continue
            if isinstance(values, list):
                param_names.append(param)
                param_values.append(values)
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        configs = []
        for combo in combinations:
            train_run_config = {
                'epochs': self.sweep['epochs'] ,
                'patience': self.sweep.get('patience', 10),
                'plot_freq': self.meta_data['plot_freq'],
                'model': self._build_model_config(self.sweep['model'] ),
                'optimizer': self._build_optimizer_config(self.sweep['optimizer'] ),
                'criterion': self._build_criterion_config(self.sweep['criterion'] ),
                'data': self._build_data_config(self.sweep['data'])
            }
            # Create sampled parameters dictionary
            sampled_params = dict(zip(param_names, combo))
            
            # Apply sampled parameters
            self._apply_sampled_params(train_run_config, sampled_params)
            
            configs.append(train_run_config)
        
        return configs
    
    def _apply_sampled_params(self, config, sampled_params):
        """Apply sampled hyperparameters to the appropriate sections of the config"""
        valid_params = {'lr', 'weight_decay', 'batch_size', 'hidden_dim', 'dropout', 'label_smoothing'}
        

        for param, value in sampled_params.items():
            if param not in valid_params:
                raise ValueError(f"Unknown hyperparameter: {param}")
            
            if param == 'lr':
                config['optimizer']['optimizer_params']['lr'] = value
            elif param == 'weight_decay':
                config['optimizer']['optimizer_params']['weight_decay'] = value
            elif param == 'batch_size':
                config['data']['batch_size'] = value
            elif param in ['hidden_dim', 'dropout']:
                config['model']['model_hyperparams'][param] = value
            elif param == 'label_smoothing':
                config['criterion']['criterion_params']['label_smoothing'] = value



    def _build_model_config(self, model_section):
        """Build model config dic"""
        config = {
            'model_path': model_section['model_path'],
            'model_hyperparams': {}
        }
        
        # Handle model weights loading
        if 'model_weights' in model_section and model_section['model_weights'] is not None and model_section['model_weights'] != '':
            config['model_weights'] = model_section['model_weights']
        else:
            config['model_weights'] = None
        
        for key, value in model_section.items():
            if key not in ['model_path', 'model_weights']:
                config['model_hyperparams'][key] = value
        
        return config
    
    def _build_optimizer_config(self, optimizer_section):
        """Build optimizer config dic"""
        config = {
            'optimizer': optimizer_section['optimizer'],
            'optimizer_params': {}
        }
        
        # Handle optimizer weights loading
        if 'optimizer_weights' in optimizer_section and optimizer_section['optimizer_weights'] is not None and optimizer_section['optimizer_weights'] != '':
            config['optimizer_weights'] = optimizer_section['optimizer_weights']
        else:
            config['optimizer_weights'] = None

        
        for key, value in optimizer_section.items():
            if key not in ['optimizer', 'optimizer_weights']:
                config['optimizer_params'][key] = value
        
        return config
    
    def _build_criterion_config(self, criterion_section):
        """Build criterion config dic"""
        config = {
            'criterion': criterion_section['criterion'],
            'criterion_params': {}
        }
        
        for key, value in criterion_section.items():
            if key != 'criterion':
                config['criterion_params'][key] = value
        
        return config
    
    def _build_data_config(self, data_section):
        return {
            'train_path': data_section['train_path'],
            'dev_path': data_section['dev_path'],
            'test_path': data_section.get('test_path', None),
            'batch_size': data_section['batch_size'],
            'use_test': data_section.get('use_test', True)
        }
    
    def get_run_on_gpu(self):
        """ returns if run_on_gpu was true in the config"""
        return self.run_on_gpu

    def get_experiment_dir(self):
        """ returns where the exper dir is located"""
        return self.expt_dir

    def get_sub_runs_list(self):
        """" return the sub runs list"""
        return self.sub_runs_list


if __name__ == "__main__":
    print('hi')