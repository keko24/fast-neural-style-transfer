import torch

class Trainer(object):
    def __init__(self, model, loss_network, optimizer, DEVICE):
        super(Trainer, self).__init__()
        self.DEVICE = DEVICE
        self.model = model
        self.optimizer = optimizer
        self.loss_network = loss_network

    def train(self, setup):
        for epoch in setup["epochs"]:
            train_loader = self.train_data()
            y_pred_total = []
            y_total = []
            train_loss = 0
            self.model.train()
            for index, (sample, target) in train_loader:
                inputs, target = sample.to(self.DEVICE), target.to(self.DEVICE)
            
                self.optimizer.zero_grad()
                y_pred = self.model(inputs)
                self.loss_network(y_pred)
                
                content_loss = torch.zeros(1, device=self.DEVICE, dtype=torch.float)
                for cl in self.loss_network.style_losses:
                    content_loss += cl
                content_loss *= setup["content_weight"]

                style_loss = torch.zeros(1, device=self.DEVICE, dtype=torch.float)
                for sl in self.loss_network.style_losses:
                    style_loss += sl
                style_loss *= setup["style_weight"]
                
                loss = content_loss + style_loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()


            
