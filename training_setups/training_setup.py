import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CustomDataset


class TrainingSetup(nn.Module):
    def __init__(self, base_model):
        super(TrainingSetup, self).__init__()
        self.base_model = base_model
        self.dataset = CustomDataset("data.csv", transform=self.base_model.transform)

    def kfold_training(self, kf):
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            train_data = torch.utils.data.Subset(self.dataset, train_idx)
            val_data = torch.utils.data.Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=self.base_model.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.base_model.batch_size)

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-6)
            criterion = nn.CosineEmbeddingLoss()

            for epoch in range(num_epochs):
                model.train()
                # Training loop here
                for batch in train_loader:
                    # Implement your training logic
                    pass

                # Validation loop here
                model.eval()
                with torch.no_grad():
                    # Implement your validation logic
                    pass

            # Save the trained model
            model_path = f"fold-{fold}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model for fold {fold} as {model_path}")
