import torch.nn as nn
from torchvision.transforms import transforms

from training_setups.training_setup import TrainingSetup
from training_setups.utils import GaussianBlur, CustomImageDataset, TwoCropsTransform


class SimSiam(TrainingSetup):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__(base_encoder)
        self.name = "SimSiam"

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        if hasattr(self.encoder, "fc"):
            prev_dim = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # first layer
                                            nn.Linear(prev_dim, prev_dim, bias=False),
                                            nn.BatchNorm1d(prev_dim),
                                            nn.ReLU(inplace=True),  # second layer
                                            self.encoder.fc,
                                            nn.BatchNorm1d(dim, affine=False))  # output layer
            self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        else:
            prev_dim = self.encoder.classifier[-1].weight.shape[1]
            self.encoder.classifier[-1] = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                                        nn.BatchNorm1d(prev_dim),
                                                        nn.ReLU(inplace=True),  # first layer
                                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                                        nn.BatchNorm1d(prev_dim),
                                                        nn.ReLU(inplace=True),  # second layer
                                                        self.encoder.classifier[-1],
                                                        nn.BatchNorm1d(dim, affine=False))  # output layer
            self.encoder.classifier[-1][6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        self.train_dataset = CustomImageDataset(df, dataset_path=dataset_path,
                                                transform=TwoCropsTransform(transforms.Compose(augmentation)))

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

    def get_features(self, image):
        z1 = self.encoder(image)
        # p1 = self.predictor(z1)
        return z1
