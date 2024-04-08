import time
from tqdm import tqdm

import torch

from loss import calculate_total_variation_loss

class Trainer:
    def __init__(self, data, model, loss_network, optimizer, setup, DEVICE) -> None:
        super(Trainer, self).__init__()
        self.DEVICE = DEVICE
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.loss_network = loss_network
        self.setup = setup

    def train(self) -> None:
        for epoch in range(self.setup["epochs"]):
            train_loader = self.data.train_data()
            train_loss = 0
            self.model.train()
            epoch_start_time = time.time()
            dataset_length = len(train_loader.dataset)
            for img_idx, sample in tqdm(enumerate(train_loader)):
                inputs = sample.to(self.DEVICE)
            
                self.optimizer.zero_grad()
                y_pred = self.model(inputs)
                self.loss_network(y_pred, inputs)
                
                content_loss = torch.zeros(1, device=self.DEVICE, dtype=torch.float)
                for cl in self.loss_network.style_losses:
                    content_loss += cl.loss
                content_loss *= self.setup["content_weight"]

                style_loss = torch.zeros(1, device=self.DEVICE, dtype=torch.float)
                for sl in self.loss_network.style_losses:
                    style_loss += sl.loss
                style_loss *= self.setup["style_weight"]
                
                tv_loss = self.setup["tv_weight"] * calculate_total_variation_loss(y_pred)

                loss = content_loss + style_loss + tv_loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if img_idx % 20 == 0:
                    print("Training Loss: {:.8f}".format(loss.item()))

            epoch_end_time = int(time.time() - epoch_start_time)
            print("Epoch: {}/{}.. ".format(epoch + 1, self.setup["epochs"]),
              "Training Loss: {:.3f}".format(train_loss))
            print('epoch {} end time: {:02d}:{:02d}:{:02d}'.format(epoch + 1, epoch_end_time // 3600, epoch_end_time % 3600 // 60, epoch_end_time % 60))
            
