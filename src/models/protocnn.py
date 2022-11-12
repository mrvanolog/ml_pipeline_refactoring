import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class ResidualBlock(nn.Module):
    """The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

        Parameters
        ----------
        in_channels : int
            The number of channels (feature maps) of the incoming embedding
        out_channels : int
            The number of channels after the first convolution
        dilation : int, optional
            Dilation rate of the first convolution, by default 1
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int=1):
        super().__init__()

        # Initialize the required layers
        self.skip = nn.Sequential()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)


class ProtCNN(pl.LightningModule):
    """ProtCNN model (https://www.biorxiv.org/content/10.1101/626507v3.full).

        Parameters
        ----------
        num_classes : int
            Number of unique classes
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            nn.Flatten(start_dim=1),
            torch.nn.Linear(7680, num_classes)
        )

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        """Method that defines one training step for pytorch lightning.
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Method that defines one validation step for pytorch lightning.
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc,
                 metric_attribute='valid_acc', on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self):
        """Method that defines optimizer configuration for pytorch lightning.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
