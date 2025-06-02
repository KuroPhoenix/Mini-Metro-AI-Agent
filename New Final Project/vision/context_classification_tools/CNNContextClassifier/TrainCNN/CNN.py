import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import pandas as pd
import csv

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be no more than 3 convolution layers
        super(CNN, self).__init__()
        self.feature_learning = nn.Sequential(
            # First
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Second
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Third
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # calculate in_features for Linear layer(only once after change implementation in FL func)
        # fl_input = torch.zeros(1, 3, 224, 224)  # input size
        # fl_output = self.feature_learning(fl_input)
        # cl_in_features = fl_output.view(-1).size(0)
        # print(f"\ncl_in: {cl_in_features}\n")

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=36864, out_features=num_classes),   # Fully connected layer # 36864 is cl_in_features
        )

    def forward(self, x):
        # (TODO) Forward the model
        x = self.feature_learning(x)
        x = self.classification(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Adam, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()   # Put model in train mode
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        # Move to device(GPU/CPU)
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)     # Pass each image through our model, and predict its label
        loss = criterion(predictions, labels)      # Compute loss of each input
        optimizer.zero_grad()       # Clear previous gradient
        loss.backward()
        # Update parameters of model
        optimizer.step()

        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)
    return avg_loss

def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()    # Put model in evaluation mode
    val_loss = 0.0
    correct = 0     # number of correctly predicted
    total = 0
    with torch.no_grad():   # Disable grad compute -> efficient
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)         # prediction is a tensor(shape = [batch_num, class_num])
            # Compute loss
            loss = criterion(predictions, labels)
            val_loss += loss.item()
            # Compute num of correct predictions in the batch
            # Firstly, determine the predicted label of each image
            _, predicted_labels = torch.max(predictions, 1)    # Find max index(class) along dimension 1(class dim)
            correct += (predicted_labels == labels).sum().item()   # Elementwise comparison -> sum up correct number
            total += labels.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    result = []
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions, 1)
            res_content = zip(image_names, predicted_labels.cpu().numpy())
            result.extend(list(res_content))
    
    with open('CNN.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['id', 'prediction'])
        writer.writerows(result)
    print(f"Predictions saved to 'CNN.csv'")
    return