from enum import Enum, auto
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

class SaveMode(Enum):
    CHECKPOINT = auto()
    JUST_MODEL = auto()

class Train_Loop():
    def __init__(self, save_dir, device='cpu', plot_freq=2, autosave_freq=10, patience=10):
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


        #plotting metrics
        self.lossi = []
        self.devlossi = []
        self.testlossi = []
        self.trainf1i = []
        self.devf1i = []
        self.testf1i = []

        self.moving_mean_lossi = []
        self.moving_mean_devlossi = []
        self.moving_mean_trainf1i = []
        self.moving_mean_devf1i = []

        self.best_loss = float('inf')
        self.best_dev_loss = float('inf')
        self.best_f1 = 0.0
        self.best_dev_f1 = 0.0

        self.best_loss_idx = None      
        self.best_dev_loss_idx = None
        self.best_f1_idx = None
        self.best_dev_f1_idx = None

        self.curr_epoch = 0 


        
        # early stopping
        self.epochs_without_improvement = 0
        self.best_model_state = None

    def train(self, model, train_loader, dev_loader, test_loader, optimizer, criterion):
        if self.has_run:
            print("ERROR: The training loop has already ruun for this object.\nYou need to re __init__ a training loop if you wish to train.\nTrainLoop obj can't be reused")
        
        self.has_run = True # to prevent for running the loop again without inilalizing the params again

        for epoch in tqdm(self.epochs):
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

            self.lossi.append(total_loss / len(train_loader)) #append avg loss over train  batches

            #DEV 
            model.eval()
            total_dev_loss = 0.0
            with torch.no_grad():
                for dev_X_batch , y_dev_batch in dev_loader:
                    dev_X_batch, dev_y_batch = dev_X_batch.to(self.device), dev_y_batch.to(self.device)
                    dev_pred = model(dev_X_batch)
                    dev_loss = criterion(dev_pred ,y_dev_batch)
                    total_dev_loss += dev_loss.item()

                self.devlossi.append(total_dev_loss / len(dev_loader)) #append avg loss over dev  batches

            #TEST
            total_test_loss = 0.0
            with torch.no_grad():
                for test_X_batch , y_test_batch in test_loader:
                    _X_batch, _y_batch = _X_batch.to(self.device), _y_batch.to(self.device)
                    test_pred = model(test_X_batch)
                    test_loss = criterion(test_pred ,y_test_batch)
                    total_test_loss += test_loss.item()

                self.devlossi.append(total_test_loss / len(test_loader)) #append avg loss over test  batches


            # UPDATES
            self._update_early_stop()

            # CHECKS
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


    def _update_early_stop(self):
        if self.devlossi[-1] < self.best_dev_loss:
            self.best_dev_loss = self.devlossi[-1]
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    #only call when need for plotting it is expensive
    def _update_plotting_metrics(self):
        
        # find best values
        best_dev_loss = min(dev_lossi)
        best_loss = min(lossi)

        best_dev_f1i= max(devf1i)
        best_f1i = max(f1i)

        # find best value idxs


        # calc moving avgs
        




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
        plt.plot(self.dev_lossi, color="#eb7734", label=f'Dev Loss; min={self.best_dev_loss:0.4f}')
        plt.plot(self.lossi_mean.squeeze(), alpha=0.55, color="#346beb");
        plt.plot(self.devlossi_mean.squeeze(), alpha=0.55, color="#eb7734"); 


        plt.scatter(self.best_loss_idx, self.best_loss, color="black", s=20, marker='v', zorder=6 )
        plt.scatter(self.best_dev_loss_idx, self.best_dev_loss, color="black", s=20, marker='v', zorder=6)


        plt.grid(True, axis='both', alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()


        plt.subplot(2, 1, 2)

        plt.plot(self.f1i, color="#346beb", label=f'Train F1; max={self.best_f1i:0.2f}' )
        plt.plot(self.devf1i, color="#eb7734", label=f'Dev F1; max={self.best_dev_f1i:0.2f}')
        plt.plot(self.f1i_mean.squeeze(), alpha=0.55, color="#346beb");
        plt.plot(self.devf1i_mean.squeeze(), alpha=0.55, color="#eb7734"); 


        plt.scatter(self.best_f1_idx, self.best_f1i, color="black", s=20, marker='v', zorder=6 )
        plt.scatter(self.best_dev_f1_idx, self.best_dev_f1i, color="black", s=20, marker='v', zorder=6)

        plt.grid(True, axis='both', alpha=0.7)
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.legend()

        plt.tight_layout()

    def _do_early_stop(self, model):
        print(f"Early stopping at epoch {self.curr_epoch}")
        self._save(model=model, mode=SaveMode.JUST_MODEL, optimizer=None)

    def _do_auto_save(self, model, optimizer):
        self.save(model=model, optimizer=optimizer, mode=SaveMode.CHECKPOINT)


    def _save(self, model, mode: SaveMode, optimizer=None):
            if mode == SaveMode.CHECKPOINT:
                checkpoint = {
                    'epoch': self.curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': self.lossi,
                    'dev_losses': self.devlossi,
                    'best_dev_loss': self.best_dev_loss
                }
                torch.save(checkpoint, f"{self.save_dir}/checkpoint_epoch_{self.curr_epoch}.pt")

            elif mode == SaveMode.JUST_MODEL:
                torch.save(model.state_dict(), f"{self.save_dir}/model_at_{self.curr_epoch}.pt")