import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from training_setups import SimSiam


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data['filename'].iloc[idx])
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load your dataset
dataset = CustomDataset("data.csv", transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf = KFold(n_splits=5, shuffle=True, random_state=42)


num_epochs = 10
batch_size = 32
learning_rate = 0.001

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = SimSiam(dim=2048, pred_dim=512).to(device)
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
