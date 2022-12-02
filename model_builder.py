from torch import nn, optim
from torchvision.models import resnet50, ResNet50_Weights


def define_layers():
    """
    Build the layers of the neural network.
    Convolutional Size =  (Size - Kernel + 2 * Padding) / Stride + 1
    Pooling Size =        (Size + 2 * Padding - Kernel) / Stride + 1
    Flatten Size =        Last Convolutional Out Channels * Size^2
    :return: A sequential layer structure which has a 48x48 single channel starting input layer and a 7 final output layer.
    """
    return small_network()
    #return large_network()
    #return resnet50_network()


def define_loss():
    """
    Choose the loss calculation method for the network.
    :return: A loss method to use.
    """
    return nn.CrossEntropyLoss()


def define_optimizer(neural_network):
    """
    Choose the optimizer method for the network.
    :param neural_network: The neural network.
    :return: An optimizer method to use.
    """
    return optim.Adam(neural_network.parameters())


def small_network():
    """
    A network composing of several, decreasing in size convolutional layers followed by two fully-connected layers.
    :return: A small CNN.
    """
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),  # (48 - 3 + 2 * 1) / 1 + 1 = 48
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (48 + 2 * 0 - 2) / 2 + 1 = 24

        nn.Conv2d(64, 64, 3, padding=1),    # (24 - 3 + 2 * 1) / 1 + 1 = 24
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (24 + 2 * 0 - 2) / 2 + 1 = 12

        nn.Conv2d(64, 64, 3, padding=1),    # (12 - 3 + 2 * 1) / 1 + 1 = 12
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (12 + 2 * 0 - 2) / 2 + 1 = 6

        nn.Conv2d(64, 64, 3, padding=1),    # (6 - 3 + 2 * 1) / 1 + 1 = 6
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (6 + 2 * 0 - 2) / 2 + 1 = 3

        nn.Flatten(),           # 64 * 3^2 = 576
        nn.Linear(576, 576),
        nn.ReLU(),
        nn.Linear(576, 7)
    )


def large_network():
    """
    Similar to the "Small" network, testing to see if simply increasing initial network input for more convolutional layers will improve accuracy.
    :return: A larger CNN.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(192),

        nn.Conv2d(1, 64, 3, padding=1),  # (192 - 3 + 2 * 1) / 1 + 1 = 192
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # (192 + 2 * 0 - 2) / 2 + 1 = 96

        nn.Conv2d(64, 64, 3, padding=1),  # (96 - 3 + 2 * 1) / 1 + 1 = 96
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # (96 + 2 * 0 - 2) / 2 + 1 = 48

        nn.Conv2d(64, 64, 3, padding=1),  # (48 - 3 + 2 * 1) / 1 + 1 = 48
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (48 + 2 * 0 - 2) / 2 + 1 = 24

        nn.Conv2d(64, 64, 3, padding=1),    # (24 - 3 + 2 * 1) / 1 + 1 = 24
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (24 + 2 * 0 - 2) / 2 + 1 = 12

        nn.Conv2d(64, 64, 3, padding=1),    # (12 - 3 + 2 * 1) / 1 + 1 = 12
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (12 + 2 * 0 - 2) / 2 + 1 = 6

        nn.Conv2d(64, 64, 3, padding=1),    # (6 - 3 + 2 * 1) / 1 + 1 = 6
        nn.ReLU(),
        nn.MaxPool2d(2, 2),     # (6 + 2 * 0 - 2) / 2 + 1 = 3

        nn.Flatten(),           # 64 * 3^2 = 576
        nn.Linear(576, 576),
        nn.ReLU(),
        nn.Linear(576, 7)
    )


def resnet50_network():
    """
    Model based on the ResNet50 architecture.
    :return: A ResNet50 based model.
    """
    # ResNet50 network with pretrained weights.
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    return nn.Sequential(

        # Prepare data to be fed into the ResNet50 model.
        nn.AdaptiveAvgPool2d(224),
        nn.Conv2d(1, 3, 3),

        net,

        # Flatten the output of the ResNet model and apply further processing.
        nn.Flatten(),

        nn.LazyLinear(64),
        nn.ReLU(),

        nn.Dropout(0.5),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Dropout(0.25),

        nn.Linear(32, 32),
        nn.ReLU(),

        nn.Dropout(0.15),

        nn.Linear(32, 7)
    )
