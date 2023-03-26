from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def define_layers(name: str):
    """
    Build the layers of the neural network.
    Convolutional Size =  (Size - Kernel + 2 * Padding) / Stride + 1
    Pooling Size =        (Size + 2 * Padding - Kernel) / Stride + 1
    Flatten Size =        Last Convolutional Out Channels * Size^2
    :param name: The name of the model architecture to use.
    :return: A sequential layer structure which has a 48x48 single channel starting input layer and a 7 final output layer.
    """
    if name == "Simple":
        return simple_network()
    if name == "ResNet":
        return resnet_network()
    raise ValueError(f"Model architecture {name} does not exist, options are Simple or ResNet.")


def simple_network():
    """
    A network composing of several, decreasing in size convolutional layers.
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

        final_layers()
    )


def resnet_network():
    """
    Model based on the ResNet18 architecture.
    :return: A ResNet18 based model.
    """
    # ResNet18 network with pretrained weights.
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    return nn.Sequential(

        # Prepare data to be fed into the ResNet18 model.
        nn.AdaptiveAvgPool2d(224),
        nn.Conv2d(1, 3, 3),

        net,

        final_layers()
    )


def final_layers():
    return nn.Sequential(

        # Flatten the output of the ResNet model and apply further processing.
        nn.Flatten(),

        nn.Dropout(),
        nn.LazyLinear(64),
        nn.ReLU(),

        nn.Dropout(),
        nn.Linear(64, 64),
        nn.ReLU(),

        nn.Dropout(),
        nn.Linear(64, 64),
        nn.ReLU(),

        nn.Linear(64, 7)
    )
