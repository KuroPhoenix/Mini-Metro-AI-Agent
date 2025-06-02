import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from loguru import logger

from CNN import CNN, test
from utils import TestDataset, load_train_dataset, load_test_dataset

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    """
    load data
    """
    logger.info("Start loading data")
    images, labels = load_train_dataset()
    images, labels = shuffle(images, labels, random_state=777)

    test_images = load_test_dataset()

    test_dataset = TestDataset(test_images)
    
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Load the saved model
    model = CNN().to(device)
    model.load_state_dict(torch.load('best_cnn_model.pth'))
    model.eval()


    """
    CNN - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test(model, test_loader, criterion, device)



if __name__ == '__main__':
    main()
