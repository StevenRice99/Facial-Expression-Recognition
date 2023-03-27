import argparse
import os
import time

import numpy
import pandas
import torch
import torchvision.utils
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


class FaceDataset(Dataset):
    """
    The datasets built from the CSV data.
    """

    def __init__(self, images, labels):
        """
        Create the dataset.
        :param images: The images parsed from the CSV.
        :param labels: The labels parsed from the CSV.
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        # Rearrange axis, so they are in proper input order and appear visually upright.
        self.images = torch.swapaxes(self.images, 1, 3)
        self.images = torch.swapaxes(self.images, 2, 3)
        self.labels = torch.tensor(labels, dtype=torch.int)
        self.transform = None

    def set_transform(self, percent: float):
        """
        Define the transform for the dataset.
        :param percent: A value between 0 and 1 for what extent to apply the transformations.
        :return: Nothing.
        """
        # If no 0, don't apply transformations.
        if percent <= 0:
            self.transform = None
            return
        self.transform = transforms.Compose([
            # Randomly flip the image.
            transforms.RandomHorizontalFlip(0.5 * percent),
            # Randomly adjust pixel values.
            transforms.ColorJitter(brightness=0.5 * percent, contrast=0.3 * percent, hue=0.3 * percent, saturation=0.3 * percent),
            # Randomly rotate the training data to add more variety.
            transforms.RandomRotation(20 * percent),
            # Randomly adjust image perspective.
            transforms.RandomPerspective(0.25 * percent),
            # Randomly zoom in on training data to add more variety.
            transforms.RandomResizedCrop(48, (1 - 0.3 * percent, 1), antialias=True)
        ])

    def __len__(self):
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get an image and its label from the dataset.
        :param idx: The index to get.
        :return: The image with transformations applied and its label.
        """
        return (self.images[idx] if self.transform is None else self.transform(self.images[idx])), self.labels[idx].type(torch.LongTensor)


class NeuralNetwork(nn.Module):
    """
    The neural network to train for the face dataset.
    """

    def __init__(self, name: str):
        """
        Set up the neural network loading in parameters defined in 'model_builder.py'.
        """
        super().__init__()
        # Load in defined parameters.
        self.layers = NeuralNetwork.define_layers(name)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        # Run on GPU if available.
        self.to(get_processing_device())

    @staticmethod
    def define_layers(name: str):
        """
        Build the layers of the neural network.
        Convolutional Size =  (Size - Kernel + 2 * Padding) / Stride + 1
        Pooling Size =        (Size + 2 * Padding - Kernel) / Stride + 1
        Flatten Size =        Last Convolutional Out Channels * Size^2
        :param name: The name of the model architecture to use.
        :return: A sequential layer structure which has a 48x48 single channel starting input layer and a 7 final output layer.
        """
        return NeuralNetwork.resnet_network() if name == "ResNet" else NeuralNetwork.simple_network()

    @staticmethod
    def simple_network():
        """
        A network composing of several, decreasing in size convolutional layers.
        :return: A simple CNN.
        """
        return nn.Sequential(
            # First convolutional layer.
            nn.Conv2d(1, 64, 3, padding=1),  # (48 - 3 + 2 * 1) / 1 + 1 = 48
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (48 + 2 * 0 - 2) / 2 + 1 = 24
            # Second convolutional layer.
            nn.Conv2d(64, 64, 3, padding=1),  # (24 - 3 + 2 * 1) / 1 + 1 = 24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (24 + 2 * 0 - 2) / 2 + 1 = 12
            # Third convolutional layer.
            nn.Conv2d(64, 64, 3, padding=1),  # (12 - 3 + 2 * 1) / 1 + 1 = 12
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (12 + 2 * 0 - 2) / 2 + 1 = 6
            # Fourth convolutional layer.
            nn.Conv2d(64, 64, 3, padding=1),  # (6 - 3 + 2 * 1) / 1 + 1 = 6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (6 + 2 * 0 - 2) / 2 + 1 = 3
            # Append the final learning layers.
            NeuralNetwork.final_layers()
        )

    @staticmethod
    def resnet_network():
        """
        Model based on the ResNet18 architecture.
        :return: A ResNet18 based model.
        """
        # ResNet18 network with pretrained weights.
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return nn.Sequential(
            # Prepare data to be fed into the ResNet18 model.
            nn.AdaptiveAvgPool2d(224),
            nn.Conv2d(1, 3, 3),
            # Pass to ResNet18.
            resnet,
            # Append the final learning layers.
            NeuralNetwork.final_layers()
        )

    @staticmethod
    def final_layers():
        """
        The final linear learning layers of the models.
        :return: Several linear learning and the final output layers.
        """
        return nn.Sequential(
            # Flatten the output of the ResNet model and apply further processing.
            nn.Flatten(),
            # Automatically handle scaling the flattened layer into 64 neurons.
            nn.Dropout(),
            nn.LazyLinear(64),
            nn.ReLU(),
            # Second final learning layer.
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # Third final learning layer.
            nn.Dropout(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # Final predictions layer.
            nn.Linear(64, 7)
        )

    def forward(self, image):
        """
        Feed forward an image into the neural network.
        :param image: The image as a proper tensor.
        :return: The final output layer from the network.
        """
        return self.layers(image)

    def predict(self, image):
        """
        Get the network's prediction for an image.
        :param image: The image as a proper tensor.
        :return: The number the network predicts for this image.
        """
        with torch.no_grad():
            # Get the highest confidence output value.
            return torch.argmax(self.forward(image), axis=-1)

    def optimize(self, image, label):
        """
        Optimize the neural network to fit the training data.
        :param image: The image as a proper tensor.
        :param label: The label of the image.
        :return: The network's loss on this prediction.
        """
        self.optimizer.zero_grad()
        loss = self.loss(self.forward(image), label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def get_processing_device():
    """
    Get the device to use for training, so we can use the GPU if CUDA is available.
    :return: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_tensor(tensor, device=get_processing_device()):
    """
    Convert an image to a tensor to run on the given device.
    :param tensor: The data to convert to a tensor.
    :param device: The device to use for training being a CUDA GPU if available, otherwise the CPU.
    :return: The data ready to be used.
    """
    return tensor.to(device)


def dataset_details(title: str, labels):
    """
    Output dataset details to the console.
    :param title: The title of the dataset to tell if this is the training or testing dataset.
    :param labels: The labels.
    :return: The total count of the dataset.
    """
    counts = [0, 0, 0, 0, 0, 0, 0]
    for label in labels:
        counts[label] += 1
    total = sum(counts)
    print(f"{title} Dataset: {total}\n"
          f"Angry:    {counts[0]:>5}\t{counts[0] / total * 100}%\n"
          f"Disgust:  {counts[1]:>5}\t{counts[1] / total * 100}%\n"
          f"Fear:     {counts[2]:>5}\t{counts[2] / total * 100}%\n"
          f"Happy:    {counts[3]:>5}\t{counts[3] / total * 100}%\n"
          f"Sad:      {counts[4]:>5}\t{counts[4] / total * 100}%\n"
          f"Surprise: {counts[5]:>5}\t{counts[5] / total * 100}%\n"
          f"Neutral:  {counts[6]:>5}\t{counts[6] / total * 100}%")
    return total


def data_image(dataloader, title: str):
    """
    Generate a sample grid image of the dataset.
    :param dataloader: A dataloader.
    :param title: The title to give the image.
    :return: Nothing.
    """
    if not os.path.exists(os.path.join(os.getcwd(), f"{title}.png")):
        torchvision.utils.save_image(torchvision.utils.make_grid(iter(dataloader).__next__()[0]), os.path.join(os.getcwd(), f"{title}.png"))


def test(model, batch: int, dataloader):
    """
    Test a neural network.
    :param model: The neural network.
    :param batch: The batch size.
    :param dataloader: The dataloader to test.
    :return: The model's accuracy.
    """
    # Switch to evaluation mode.
    model.eval()
    # Count how many are correct.
    correct = 0
    # Loop through all data.
    for image, label in dataloader:
        # If properly predicted, count it as correct.
        correct += (to_tensor(label) == model.predict(to_tensor(image))).sum()
    # Calculate the overall accuracy.
    return correct / (len(dataloader) * batch) * 100


def prepare_data(data):
    """
    Convert the raw pixel data strings for use with PyTorch.
    :param data: The data frame loaded from the CSV with emotions and pixel data.
    :return: Images and labels ready for PyTorch.
    """
    images = numpy.zeros(shape=(len(data), 48, 48))
    # Break apart the string and resize it as a 48x48 image.
    for i, row in enumerate(data.index):
        image = numpy.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = numpy.reshape(image, (48, 48))
        images[i] = image
    # Ensure single channel added for proper inputs.
    images = images.reshape((images.shape[0], 48, 48, 1))
    # Scale all color values between 0 and 1.
    images = images.astype('float32') / 255
    return images, numpy.array(list(map(int, data['emotion'])))


def save(name: str, mode: str, model, best_model, epoch: int, best_accuracy: float, loss: float):
    """
    Save to a PT file.
    :param name: The name of the model architecture.
    :param mode: The name of the training mode.
    :param model: The model.
    :param best_model: The best model state dict.
    :param epoch: The current training epoch.
    :param best_accuracy: The best accuracy that has been reached.
    :param loss: The last training loss.
    :return: Nothing.
    """
    torch.save({
        'Best': best_model,
        'Training': model.state_dict(),
        'Optimizer': model.optimizer.state_dict(),
        'Epoch': epoch,
        'Best Accuracy': best_accuracy,
        'Loss': loss
    }, f"{os.getcwd()}/Models/{name}-{mode}.pt")


def write_parameters(name: str, mode: str, best_accuracy: float, train_accuracy: float, inference_time: float, trainable_parameters: int, best_epoch: int):
    """
    Write key parameters to a text file.
    :param name: The name of the model architecture.
    :param mode: The name of the training mode.
    :param best_accuracy: The best accuracy that has been reached.
    :param train_accuracy: The training accuracy that has been reached.
    :param inference_time: The average inference time.
    :param trainable_parameters: The number of trainable parameters in the network.
    :param best_epoch: The epoch which the best accuracy was reached on.
    """
    parameters = open(f"{os.getcwd()}/Models/{name}-{mode}.txt", "w")
    parameters.write(f"Testing Accuracy: {best_accuracy}\n"
                     f"Training Accuracy: {train_accuracy}\n"
                     f"Average Inference Time: {inference_time} ms\n"
                     f"Trainable Parameters: {trainable_parameters}\n"
                     f"Best Epoch: {best_epoch}")
    parameters.close()


def main(epochs: int, batch: int):
    """
    Main program execution.
    :param epochs: The number of epochs to train for.
    :param batch: The batch size.
    :return: Nothing.
    """
    print(f"Face Expression Recognition Deep Learning")
    print(f"Running on GPU with CUDA {torch.version.cuda}." if torch.cuda.is_available() else "Running on CPU.")
    if not os.path.exists(os.path.join(os.getcwd(), "Data.csv")):
        print("Data.csv missing, visit https://github.com/StevenRice99/Facial-Expression-Recognition#setup for instructions.")
        return
    # Setup datasets.
    print("Loading data...")
    df = pandas.read_csv(f"{os.getcwd()}/Data.csv")
    train_images, train_labels = prepare_data(df[df['Usage'] == 'Training'])
    test_images, test_labels = prepare_data(df[df['Usage'] != 'Training'])
    training_data = FaceDataset(train_images, train_labels)
    testing_data = FaceDataset(test_images, test_labels)
    training_total = dataset_details("Training", train_labels)
    testing_total = dataset_details("Testing", test_labels)
    # Generate sample images of the data.
    testing = DataLoader(testing_data, batch_size=batch, shuffle=True)
    training_data.set_transform(1)
    training = DataLoader(training_data, batch_size=batch, shuffle=True)
    data_image(training, 'Augmented')
    data_image(testing, 'Normal')
    if batch < 1:
        batch = 1
    # Train models.
    for name in ["Simple", "ResNet"]:
        # Train each model on every training mode.
        for mode in ["Normal", "Augmented", "Gradual"]:
            # Check if an existing model for this mode exists.
            if os.path.exists(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.pt")):
                try:
                    saved = torch.load(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.pt"))
                    epoch = saved['Epoch']
                    best_accuracy = saved['Best Accuracy']
                    # If already done training this joint, skip to the next.
                    if epoch >= epochs:
                        print(f"{name} | {mode} | Accuracy = {best_accuracy}%")
                        continue
                    model = NeuralNetwork(name)
                    model.load_state_dict(saved['Training'])
                    model.optimizer.load_state_dict(saved['Optimizer'])
                    best_model = saved['Best']
                    loss = saved['Loss']
                    print(f"Continuing training for {name} on mode {mode} from epoch {epoch} with batch size {batch} for {epochs} epochs.")
                except:
                    print("Unable to load training data, exiting.")
                    return
            # Create a new model if none already exists.
            else:
                model = NeuralNetwork(name)
                best_model = model.state_dict()
                loss = -1
                epoch = 1
                print(f"Starting training for {name} on mode {mode} with batch size {batch} for {epochs} epochs.")
            # Ensure folder to save models exists.
            if not os.path.exists(os.path.join(os.getcwd(), "Models")):
                os.mkdir(os.path.join(os.getcwd(), "Models"))
            # Test the model.
            start = time.time_ns()
            accuracy = test(model, batch, testing)
            end = time.time_ns()
            inference_time = ((end - start) / testing_total) / 1e+6
            # If new training, write header for new CSV file.
            if epoch == 1:
                best_accuracy = accuracy
                f = open(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.csv"), "w")
                f.write("Epoch,Loss,Accuracy,Best Accuracy")
                f.close()
            train_accuracy = test(model, batch, training)
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            write_parameters(name, mode, best_accuracy, train_accuracy, inference_time, trainable_parameters, 0)
            save(name, mode, model, best_model, epoch, best_accuracy, loss)
            # If not on gradual mode, set the training data as it does not change.
            if mode != "Gradual":
                training_data.set_transform(-1 if mode == "Normal" else 1)
                training = DataLoader(training_data, batch_size=batch, shuffle=True)
            # Loop training.
            while True:
                # Output final result.
                if epoch > epochs:
                    print(f"{name} | {mode} | Accuracy = {best_accuracy}%")
                    break
                loss_message = "Loss = " + (f"{loss:.4}" if epoch > 1 else "N/A")
                msg = f"{name} | {mode} | Epoch {epoch}/{epochs} | {loss_message} | Accuracy = {accuracy:.4}% | Best = {best_accuracy:.4}%"
                # Reset loss every epoch.
                loss = 0
                # Switch to training mode.
                model.train()
                # If on gradual mode, increment the data augmentation.
                if mode == "Gradual":
                    training_data.set_transform((epoch - 1) / float(epochs - 1))
                    training = DataLoader(training_data, batch_size=batch, shuffle=True)
                # Train on the training data.
                for image, label in tqdm(training, msg):
                    loss += model.optimize(to_tensor(image), to_tensor(label))
                loss /= training_total
                # Check how well the newest epoch performs.
                start = time.time_ns()
                accuracy = test(model, batch, testing)
                end = time.time_ns()
                # Check if this is the new best model.
                if accuracy > best_accuracy:
                    best_model = model.state_dict()
                    best_accuracy = accuracy
                    inference_time = ((end - start) / testing_total) / 1e+6
                    train_accuracy = test(model, batch, training)
                    write_parameters(name, mode, best_accuracy, train_accuracy, inference_time, trainable_parameters, epoch)
                # Save data.
                f = open(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.csv"), "a")
                f.write(f"\n{epoch},{loss},{accuracy},{best_accuracy}")
                f.close()
                epoch += 1
                save(name, mode, model, best_model, epoch, best_accuracy, loss)


if __name__ == '__main__':
    try:
        desc = "Face Expression Recognition Deep Learning\n-----------------------------------------"
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
        parser.add_argument("-e", "--epoch", type=int, help="The number of epochs to train for.", default=100)
        parser.add_argument("-b", "--batch", type=int, help="Training and testing batch size.", default=64)
        a = vars(parser.parse_args())
        main(a["epoch"], a["batch"])
    except KeyboardInterrupt:
        print("Training Stopped.")
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Try running with a smaller batch size.")
    except ValueError as error:
        print(error)
