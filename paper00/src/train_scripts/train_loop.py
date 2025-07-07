from enum import Enum, auto
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os
import json
from torchmetrics.classification import BinaryF1Score

class SaveMode(Enum):
    CHECKPOINT = auto()
    JUST_MODEL_AND_METRICS = auto()

class Train_Loop():
    def __init__(self, save_dir, epochs, device='cpu', plot_freq=3, autosave_freq=10, patience=10):
        """
        Initializes a new TrainLoop instance.

        Args:
            device: Device to train on (ex. 'gpu:0','cpu', 'mps')
            plot_freq (int): After how many epochs to plot 
            autosave_freq (int): after how many epochs to save automatically
            patience (int): dev set loss patience

        """
        self.has_run = False
        self.device = device
        self.plot_freq = plot_freq 
        self.autosave_freq = autosave_freq 
        self.patience = patience
        self.save_dir = save_dir
        self.epochs = epochs
        self.curr_epoch = 0 

        os.mkdir(self.save_dir)


        #plotting metrics
        self.lossi = []
        self.devlossi = []
        self.testlossi = []
        self.f1i = []
        self.devf1i = []
        self.testf1i = []

        self.moving_mean_lossi = torch.tensor([]) 
        self.moving_mean_devlossi = torch.tensor([]) 
        self.moving_mean_f1i = torch.tensor([]) 
        self.moving_mean_devf1i = torch.tensor([]) 

        self.best_loss = float('inf')
        self.best_dev_loss = float('inf')
        self.best_f1 = 0.0
        self.best_dev_f1 = 0.0

        self.best_loss_idx = None      
        self.best_dev_loss_idx = None
        self.best_f1_idx = None
        self.best_dev_f1_idx = None


        # early stopping
        self.epochs_without_improvement = 0
        # self.best_model_state = None


        self.train_f1 = BinaryF1Score().to(device)
        self.dev_f1 = BinaryF1Score().to(device)
        self.test_f1 = BinaryF1Score().to(device)


    def train(self, model, optimizer, criterion, train_loader, dev_loader, test_loader=None):
        assert not self.has_run, "Training loop already executed. Create new instance."
        assert self.epochs > 0, "epochs must be positive"
        
        self.has_run = True # to prevent for running the loop again without inilalizing the params again
        model = model.to(self.device)

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            self.curr_epoch = epoch + 1
            
            #TRAIN
            model.train()
            total_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                self.train_f1(pred, y_batch) #we assume (Batch x Seq Len)

            self.lossi.append(total_loss / len(train_loader)) #append avg loss over train  batches
            self.f1i.append(self.train_f1.compute().item()) #append  f1 over train  batches
            self.train_f1.reset()

            #DEV 
            model.eval()
            total_dev_loss = 0.0
            with torch.no_grad():
                for dev_X_batch , dev_y_batch in dev_loader:
                    dev_X_batch, dev_y_batch = dev_X_batch.to(self.device), dev_y_batch.to(self.device)
                    dev_pred = model(dev_X_batch)
                    dev_loss = criterion(dev_pred, dev_y_batch)
                    total_dev_loss += dev_loss.item()
                    self.dev_f1(dev_pred, dev_y_batch) #we assume (Batch x Seq Len)


                self.devlossi.append(total_dev_loss / len(dev_loader)) #append avg loss over dev  batches
                self.devf1i.append(self.dev_f1.compute().item()) #append  f1 over dev  batches
                self.dev_f1.reset()


            # TEST
            if test_loader:
                total_test_loss = 0.0
                with torch.no_grad():
                    for test_X_batch , test_y_batch in test_loader:
                        test_X_batch, test_y_batch = test_X_batch.to(self.device), test_y_batch.to(self.device)
                        test_pred = model(test_X_batch)
                        test_loss = criterion(test_pred, test_y_batch)
                        total_test_loss += test_loss.item()
                        self.test_f1(test_pred, test_y_batch) #we assume (Batch x Seq Len)


                    self.testlossi.append(total_test_loss / len(test_loader)) #append avg loss over test  batches
                    self.testf1i.append(self.test_f1.compute().item()) #append  f1 over test  batches
                    self.test_f1.reset()


            # UPDATES
            self._update_best_metrics()
            self._update_early_stop()

            pbar.set_description(f"Device:{self.device} curr Loss: {self.lossi[-1]:.4f}, curr Dev Loss: {self.devlossi[-1]:.4f}")

            # CHECKS
            if self._is_best():
                self._save(model=model, mode=SaveMode.JUST_MODEL_AND_METRICS, name='best_dev_')

            if self._should_plot():
                self._do_plot()


            if self._should_early_stop():
                self._do_early_stop(model=model)
                self._do_plot()
                break

            if self._should_autosave():
                self._do_auto_save(model=model, optimizer=optimizer)

            
        

        #PLOT ON END
        self._do_plot()

    def _update_best_metrics(self):
        # Update best values
        if self.lossi[-1] < self.best_loss:
            self.best_loss = self.lossi[-1]
            self.best_loss_idx = len(self.lossi) - 1
            
        if self.devlossi[-1] < self.best_dev_loss:
            self.best_dev_loss = self.devlossi[-1]
            self.best_dev_loss_idx = len(self.devlossi) - 1
            
        if self.f1i[-1] > self.best_f1:
            self.best_f1 = self.f1i[-1]
            self.best_f1_idx = len(self.f1i) - 1
            
        if self.devf1i[-1] > self.best_dev_f1:
            self.best_dev_f1 = self.devf1i[-1]
            self.best_dev_f1_idx = len(self.devf1i) - 1


    def _update_early_stop(self):
        if self.devlossi[-1] == self.best_dev_loss: #we assume if == then its becasue it was jsut updated to be best metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    #only call when need for plotting it is expensive
    def _update_plotting_metrics(self, kernel_size = 3):
        
        if self.curr_epoch >= kernel_size:
            # calc moving avgs
            lossi_tensor = torch.tensor(self.lossi, dtype=torch.float32)
            devlossi_tensor = torch.tensor(self.devlossi, dtype=torch.float32)
            f1i_tensor = torch.tensor(self.f1i, dtype=torch.float32)
            devf1i_tensor = torch.tensor(self.devf1i, dtype=torch.float32)

            self.moving_mean_lossi =  torch.nn.functional.avg_pool1d(lossi_tensor.unsqueeze(dim=0), kernel_size=kernel_size, stride=1).squeeze()
            self.moving_mean_devlossi = torch.nn.functional.avg_pool1d(devlossi_tensor.unsqueeze(dim=0), kernel_size=kernel_size, stride=1).squeeze()
            self.moving_mean_f1i = torch.nn.functional.avg_pool1d(f1i_tensor.unsqueeze(dim=0), kernel_size=kernel_size, stride=1).squeeze()
            self.moving_mean_devf1i = torch.nn.functional.avg_pool1d(devf1i_tensor.unsqueeze(dim=0), kernel_size=kernel_size, stride=1).squeeze()

    def _is_best(self):
        return self.devlossi[-1] == self.best_dev_loss 

    def _should_early_stop(self):
        return self.epochs_without_improvement > self.patience

    def _should_plot(self):
        return (self.curr_epoch % self.plot_freq == 0) or (self.curr_epoch == 2)

    def _should_autosave(self):
        return (self.curr_epoch % self.autosave_freq == 0)


    def _do_plot(self):
        self._update_plotting_metrics()

        plt.figure(figsize=(12,10))
        plt.subplot(2, 1, 1)

        plt.plot(self.lossi, color="#346beb", label=f'Train Loss; min={self.best_loss:0.4f}' )
        plt.plot(self.devlossi, color="#eb7734", label=f'Dev Loss; min={self.best_dev_loss:0.4f}')
        plt.plot(self.moving_mean_lossi, alpha=0.55, color="#346beb");
        plt.plot(self.moving_mean_devlossi, alpha=0.55, color="#eb7734"); 


        plt.scatter(self.best_loss_idx, self.best_loss, color="black", s=20, marker='v', zorder=6 )
        plt.scatter(self.best_dev_loss_idx, self.best_dev_loss, color="black", s=20, marker='v', zorder=6)


        plt.grid(True, axis='both', alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()


        plt.subplot(2, 1, 2)

        plt.plot(self.f1i, color="#346beb", label=f'Train F1; max={self.best_f1:0.2f}' )
        plt.plot(self.devf1i, color="#eb7734", label=f'Dev F1; max={self.best_dev_f1:0.2f}')
        plt.plot(self.moving_mean_f1i, alpha=0.55, color="#346beb");
        plt.plot(self.moving_mean_devf1i , alpha=0.55, color="#eb7734"); 


        plt.scatter(self.best_f1_idx, self.best_f1, color="black", s=20, marker='v', zorder=6 )
        plt.scatter(self.best_dev_f1_idx, self.best_dev_f1, color="black", s=20, marker='v', zorder=6)

        plt.grid(True, axis='both', alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,f"metrics.png"))
        plt.close()

    def _do_early_stop(self, model):
        print(f"{self.device} Early stopping at epoch {self.curr_epoch}")
        self._save(model=model, mode=SaveMode.JUST_MODEL_AND_METRICS, optimizer=None, name=f'earlystop_{self.curr_epoch}_')

    def _do_auto_save(self, model, optimizer):
        self._save(model=model, optimizer=optimizer, mode=SaveMode.CHECKPOINT, name='autosave_')

    def _save(self, model, mode: SaveMode, optimizer=None, name=''):
            if mode == SaveMode.CHECKPOINT:
                checkpoint = {
                    'epoch': self.curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lossi': self.lossi,
                    'devlossi': self.devlossi,
                    'testlossi': self.testlossi,
                    'f1i': self.f1i,
                    'devf1i': self.devf1i,
                    'testf1i': self.testf1i
                }
                torch.save(checkpoint, f"{self.save_dir}/{name}checkpoint.pt")

            elif mode == SaveMode.JUST_MODEL_AND_METRICS:
                torch.save(model.state_dict(), f"{self.save_dir}/{name}model.pt")
                metrics = {
                    'epoch': self.curr_epoch,
                    'lossi': self.lossi,
                    'devlossi': self.devlossi,
                    'testlossi': self.testlossi,
                    'f1i': self.f1i,
                    'devf1i': self.devf1i,
                    'testf1i': self.testf1i
                }
                with open(f"{self.save_dir}/{name}metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)