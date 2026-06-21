import torch
import torch.nn as nn
import torch.nn.functional as F


# class structures taken from: https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch

class ImageClassifier(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CNN(ImageClassifier):
    def __init__(self, dropout_p: float=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        # Switch for stochastic (MC dropout) inference. When True, calling
        # .eval() keeps the dropout layers active while BatchNorm still uses
        # its running statistics. See enable_mc_dropout() / train().
        self.mc_dropout = False

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # self._dropout2DLayer(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: (64, 16, 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # self._dropout2DLayer(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: (128, 8, 8)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # self._dropout2DLayer(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: (256, 4, 4)

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            self._dropoutLayer(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            self._dropoutLayer(),
            nn.Linear(512, 10)
        )

    def _dropout2DLayer(self):
        if self.dropout_p > 0.0:
            return nn.Dropout2d(p=self.dropout_p)
        else:
            return nn.Identity()
    
    def _dropoutLayer(self):
        if self.dropout_p > 0.0:
            return nn.Dropout(p=self.dropout_p)
        else:
            return nn.Identity()

    def enable_mc_dropout(self, enable: bool = True):
        """Flip the stochastic-inference switch.

        When enabled, a subsequent call to .eval() leaves the dropout layers
        in training mode (so each forward pass is stochastic) while every other
        layer -- crucially BatchNorm -- still runs in eval mode using its
        running statistics.
        """
        self.mc_dropout = enable
        # Re-apply the mode so the switch takes effect immediately.
        return self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # In eval mode with MC dropout enabled, re-activate only the dropout
        # layers; BatchNorm (and everything else) stays in eval mode.
        if not mode and self.mc_dropout:
            for m in self.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                    m.train()
        return self

    def forward(self, x):
        return self.network(x)
