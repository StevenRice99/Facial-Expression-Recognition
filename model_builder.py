from torch import nn, optim


def define_layers():
    """
    Build the layers of the neural network.
    :return: A sequential layer structure which has a 48x48 starting input layer and a 7 final output layer.
    """
    # Convolutional Size =  (Size - Kernel + 2 * Padding) / Stride + 1
    # Pooling Size =        (Size + 2 * Padding - Kernel) / Stride + 1
    # Flatten Size =        Last Convolutional Out Channels * Size^2
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
