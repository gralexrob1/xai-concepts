import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


class ResnetPredictor(nn.Module):
    def __init__(self):
        super(ResnetPredictor, self).__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        self.resnet.fc.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.softmax(x, dim=1)
        return x


def train_epoch(data_loader, model, criterion, optimizer, device):

    model.train()
    losses = []
    true_labels = []
    pred_labels = []

    data_iterator = tqdm(data_loader, desc="Training")
    for img_batch, label_batch in data_iterator:

        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()
        output_batch = model(img_batch)
        loss = criterion(output_batch, label_batch)
        loss.backward()
        optimizer.step()

        true_label_batch = torch.argmax(label_batch, 1)
        pred_label_batch = torch.argmax(output_batch, 1)

        losses.append(loss.item())
        true_labels.extend(true_label_batch.cpu().tolist())
        pred_labels.extend(pred_label_batch.cpu().tolist())

        data_iterator.set_postfix(loss=np.mean(losses))

    test_macro = f1_score(true_labels, pred_labels, average="macro")
    test_micro = f1_score(true_labels, pred_labels, average="micro")

    return np.array(losses), test_macro, test_micro


def val_epoch(data_loader, model, criterion, device):

    model.eval()
    losses = []
    true_labels = []
    pred_labels = []

    data_iterator = tqdm(data_loader, desc="Validation")
    for img_batch, label_batch in data_iterator:

        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        with torch.no_grad():
            output_batch = model(img_batch)
            loss = criterion(output_batch, label_batch)

        true_label_batch = torch.argmax(label_batch, 1)
        pred_label_batch = torch.argmax(output_batch, 1)

        losses.append(loss.item())
        true_labels.extend(true_label_batch.cpu().tolist())
        pred_labels.extend(pred_label_batch.cpu().tolist())

        data_iterator.set_postfix(loss=np.mean(losses))

    test_macro = f1_score(true_labels, pred_labels, average="macro")
    test_micro = f1_score(true_labels, pred_labels, average="micro")

    return np.array(losses), test_macro, test_micro
