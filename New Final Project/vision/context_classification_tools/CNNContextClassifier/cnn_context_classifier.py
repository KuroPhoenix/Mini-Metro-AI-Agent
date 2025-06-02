
import torch
from loguru import logger
import time
from CNN import CNN
from torchvision import transforms
import pyautogui

def cnn_context_classifier(model: CNN):
    frame = pyautogui.screenshot()
    # Crop the image. Remember PIL.Image.crop takes (left, upper, right, lower)
    # Your values (0, 70, 1277, 948) seem to define the region correctly.
    image = frame.crop((0, 70, 1277, 948))

    t = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # If your CNN expects 3 channels
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    # Apply the transformations and store the result in a new variable
    # This 'processed_image' will be a PyTorch Tensor
    processed_image = t(image)

    # Add a batch dimension to the processed image
    # A single image tensor will be (C, H, W), but the model expects (N, C, H, W)
    input_tensor = processed_image.unsqueeze(0)

    # Perform inference
    with torch.no_grad(): # Disable gradient calculation for inference
        prediction = model(input_tensor)
        _, predicted_labels = torch.max(prediction, 1)
        contextID = predicted_labels.cpu().numpy()
    print(contextID[0])
    """
    1: gameplay
    2: 1-reward
    3: 2-reward
    4: gameover
    """

def load_cnn():
    # Instantiate your CNN model
    model = CNN()

    # Load the saved model weights
    try:
        model.load_state_dict(torch.load('cnn_context_classification_model.pth'))
    except FileNotFoundError:
        logger.error("best_cnn_model.pth not found. Make sure the model is trained and saved.")
        return

    # Set the model to evaluation mode
    model.eval()

    return model


if __name__ == '__main__':
    m = load_cnn()
    while True:
        cnn_context_classifier(model=m)
        time.sleep(0.1)