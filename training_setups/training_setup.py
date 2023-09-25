import torch
from torch import nn

from dataset import CustomDataset


class TrainingSetup(nn.Module):
    def __init__(self, base_model):
        super(TrainingSetup, self).__init__()
        self.base_model = base_model
        self.dataset = CustomDataset("data.csv", transform=self.base_model.transform)

    def kfold_training(self, kf):
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_data = torch.utils.data.Subset(dataset, train_idx)
            val_data = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size)
