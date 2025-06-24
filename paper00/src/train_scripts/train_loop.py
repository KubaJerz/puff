from tqdm import tqdm
import torch

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

        
        self.lossi = []
        self.devlossi = []
        self.metrics = []
        self.devmetrics = []
        self.curr_epoch = 0 


        
        # early stopping
        self.best_dev_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None

    def train(self, model, train_loader, dev_loader, optimizer, criterion):
        if self.has_run:
            print("ERROR: The training loop has already ruun for this object.\nYou need to re __init__ a training loop if you wish to train.\nTrainLoop obj can't be reused")
        
        self.has_run = True # to prevent for running the loop again without inilalizing the params again

        for epoch in tqdm(self.epochs):
            self.curr_epoch = epoch + 1
            
            #TRAIN
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.lossi.append(total_loss / len(train_loader)) #append avg loss over train  batches

            #DEV 
            model.eval()
            total_dev_loss = 0
            with torch.no_grad():
                for dev_X_batch , y_dev_batch in dev_loader:
                    dev_pred = model(dev_X_batch)
                    dev_loss = criterion(dev_pred ,y_dev_batch)
                    total_dev_loss += dev_loss.item()

                self.devlossi.append(total_dev_loss / len(dev_loader)) #append avg loss over dev  batches


            #UPDATES
            self._update_early_stop()

            #EXTRA CHECKS
            self._check_for_plot()

            self._check_for_early_stop()

            self._check_for_autosave()
        

    def _update_early_stop(self):
        if self.devlossi[-1] < self.best_dev_loss:
            self.best_dev_loss = self.devlossi[-1]
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

    def _check_for_early_stop(self):
        if self.epochs_without_improvement > self.patience:
            print(f"Early stopping at epoch {self.curr_epoch}")
            self.save_model()
            self.exit()

    def _check_for_plot(self):
        if (self.curr_epoch % self.plot_freq == 0) or (self.curr_epoch == 2):
            self.plot()

    def _check_for_autosave(self):
        if self.curr_epoch % self.autosave_freq == 0:
            self.save_model()


    def _plot(self):
        pass

    def exit(self):
        pass

    def save_model(self):
        pass

            


