from sklearn.model_selection import KFold

from models import *
from training_setups import *

# --- These are all models and training setups. They are still not all implemented
# models = [ResNet(50), "VGG-16", "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3",
#           "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7"]
#
# training_setups = ["Classification", SimSiam, "SimCLR", "BYOL", "MoCov1", "MoCov2", "MoCov3", "MAE", "I-JEPA"]

models = [ResNet]
training_setups = [SimSiam]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Running k-fold trainings...")
    for model in models:
        print(f"Model: {model}")
        for training_setup in training_setups:
            ts = training_setup(model)
            print(f"Training setup: {ts.name}")
            print("Training setup initialized!")
            ts.kfold_training(kf)


if __name__ == "__main__":
    main()
