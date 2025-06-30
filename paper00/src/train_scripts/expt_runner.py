import torch
import threading
import multiprocessing as mp
from train_loop import Train_Loop
import os
import importlib
import sys

class Expt_Runner():
    def __init__(self, expt_dir, sub_runs_list, run_on_gpu):
        self.expt_dir = expt_dir
        self.sub_runs_list = sub_runs_list.copy()
        self.run_on_gpu = run_on_gpu
        self.available_gpu_list = []

        if run_on_gpu:
            self._check_gpu_availability()

    def run(self):
        if not self.sub_runs_list:
            print("No experiments to run. 'sub_runs_list' is empty")
            return
        elif self.run_on_gpu:
            if not self.available_gpu_list:
                print("Can't run on GPU.\nNo GPU's availabe.")
                return
            self._gpu_run()
        else: 
            self._cpu_run()

    def _gpu_run(self):
        """
            NOTE: We are waiting for ALL processes to complete before starting new ones
            So if 2 GPUS they fully train then we do the next two.
        """
    
        mp.set_start_method('spawn') #, force=True) #This atart method works better with CUDA

        processes = []
        run_id = 0
        while self.sub_runs_list:
            while self.available_gpu_list and self.sub_runs_list:
                gpu_idx, name = self.available_gpu_list.pop(0)
                train_run_config = self.sub_runs_list.pop(0)

                train_loop =  Train_Loop(save_dir=f'{self.expt_dir}/{run_id}', epochs=train_run_config['epochs'], device=f'cuda:{gpu_idx}', plot_freq=train_run_config['plot_freq'])
                model, optimizer, criterion, train_loader, dev_loader, test_loader = self._setup_train_objs(train_run_config)

                p = mp.Process(target=train_loop.train, args=(model, optimizer, criterion, train_loader, dev_loader, test_loader))
                p.start()
                processes.append((p, gpu_idx, name))
                run_id += 1

            for p, gpu_idx, name in processes:
                p.join()
                self.available_gpu_list.append((gpu_idx, name))

            processes = []

    def _cpu_run(self):
        for i, train_run_config in enumerate(self.sub_runs_list):
            train_loop =  Train_Loop(save_dir=f'{self.expt_dir}/{i}', epochs=train_run_config['epochs'], device='cpu', plot_freq=10)
            model, optimizer, criterion, train_loader, dev_loader, test_loader = self._setup_train_objs(train_run_config)
            train_loop.train(model, optimizer, criterion, train_loader, dev_loader, test_loader)


    def _check_gpu_availability(self):
        if torch.cuda.is_available():
            print(f"Number of GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                self.available_gpu_list.append((i, torch.cuda.get_device_name(i)))
        else:
            print("CUDA not available")

    def _load_class_from_str(self, path_str):
        """
        take something like 'torch.nn.CrossEntropyLoss' and returns the class obj
        """
        module_path, class_name = path_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _setup_train_objs(self, train_run_config):
        model = self._setup_model(train_run_config["model"])
        optimizer = self._setup_optimizer(train_run_config["optimizer"], model.parameters())
        criterion = self._setup_criterion(train_run_config["criterion"])
        train_loader, dev_loader, test_loader = self._setup_loaders(train_run_config["data"])

        return model, optimizer, criterion, train_loader, dev_loader, test_loader



    def _setup_model(self, model_config):
        model_path = model_config.get("model_path")
        model_params = model_config.get("model_hyperparams", {})

        # If loading from .py file
        if model_path.endswith(".py"):
            module_dir = os.path.dirname(model_path)
            module_name = os.path.basename(model_path).replace(".py", "")
            sys.path.insert(0, module_dir)
            module = importlib.import_module(module_name)
            model_class = getattr(module, "Model")
            return model_class(**model_params)
        else:
            # If specifying full module path to model class
            model_class = self._load_class_from_str(model_path)
            return model_class(**model_params)


    def _setup_optimizer(self, opt_config, model_params):
        opt_class_str = opt_config["optimizer"]
        opt_params = opt_config.get("optimizer_params", {})
        opt_class = self._load_class_from_str(opt_class_str)
        return opt_class(model_params, **opt_params)


    def _setup_criterion(self, crit_config):
        crit_class_str = crit_config["criterion"]
        crit_params = crit_config.get("criterion_params", {})

        if '/' in crit_class_str: # if specified liek this: /home/kuba/projects/puff/test/loss.DiceBCELoss
            module_dir = os.path.dirname(crit_class_str)
            module_name, loss_class_name = os.path.basename(crit_class_str).rsplit('.')
            sys.path.insert(0, module_dir)
            module = importlib.import_module(module_name)
            crit_class = getattr(module, loss_class_name)
        else:
            crit_class = self._load_class_from_str(crit_class_str)
        return crit_class(**crit_params)


    def _setup_loaders(self, data_config):
        batch_size = data_config.get("batch_size")

        def load_dataset_from_dir(path, type):
            X = torch.load(os.path.join(path, f"X_{type}.pt"))
            y = torch.load(os.path.join(path, f"y_{type}.pt"))
            dataset = torch.utils.data.TensorDataset(X, y)
            return dataset

        train_dataset = load_dataset_from_dir(data_config["train_path"], type="train")
        dev_dataset = load_dataset_from_dir(data_config["dev_path"], type="dev")
        test_dataset = load_dataset_from_dir(data_config["test_path"], type="test")

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, dev_loader, test_loader
        

if __name__ == '__main__':
    expt_runner = Expt_Runner(sub_runs_list=None)
    print(expt_runner.available_gpu_list)