import torch.nn as nn

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

