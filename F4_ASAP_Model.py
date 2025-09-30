import torch.nn as nn
import torchvision.models as models

class StomataNet(nn.Module):
    def __init__(self):
        super(StomataNet, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.features = base
        ## Classification: NC or aperture
        self.classifier = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        ## Regression: aperture value
        self.regressor = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, x):
        feat = self.features(x).squeeze()
        cls_logit = self.classifier(feat)
        reg_value = self.regressor(feat)
        return cls_logit, reg_value

