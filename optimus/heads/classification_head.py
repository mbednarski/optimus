import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, model_size, n_classes):
        super().__init__()

        self.fc = nn.Linear(model_size, n_classes)

    def forward(self, x):
        # batch size x seq len x model size

        # cls: bs x  model size
        cls_token = x[:, 0, :]

        # bs x n_classes
        return self.fc(cls_token)
