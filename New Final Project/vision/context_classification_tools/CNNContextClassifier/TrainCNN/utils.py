from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # slight augmentation
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name  # Return base_name as a string, not a list
    
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    images = []
    labels = []
    name_to_label = {"1gameplay": 1, "2select_weekly_reward_1": 2, "3select_weekly_reward_2": 3, "4game_over": 4}

    for subdir in os.listdir(path):
        if subdir.startswith('.') or subdir not in name_to_label:  # Skip hidden files(for macOS)
            continue
        label = name_to_label[subdir]
        subdir_path = os.path.join(path, subdir)
        for image_file in os.listdir(subdir_path):
            if image_file.startswith('.'):
                continue
            images.append(os.path.join(subdir_path, image_file))  # Append full path
            labels.append(label)
    return images, labels

def load_test_dataset(path: str='data/test/')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for image_file in os.listdir(path):
        if image_file.lower().endswith(valid_exts):
            images.append(os.path.join(path, image_file))
    return images

def plot(train_losses: List, val_losses: List):
    # (TODO)Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    plt.figure(figsize=(10, 5))

    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xticks(epochs)      # tick to integer
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.title('loss.png')
    plt.savefig('loss.png')
    plt.close()
    print("Save the plot to 'loss.png'")
    return