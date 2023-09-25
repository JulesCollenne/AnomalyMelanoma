import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from dataset import CustomDataset
from models import *
from training_setups import *

models = [ResNet(50), "VGG-16", "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
          "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7"]

training_setups = ["Classification", SimSiam, "SimCLR", "BYOL", "MoCov1", "MoCov2", "MoCov3", "MAE", "I-JEPA"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model in models:
        for training_setup in training_setups:
            training_setup = training_setup(model)
            training_setup.kfold_train(kf)


if __name__ == "__main__":
    main()
