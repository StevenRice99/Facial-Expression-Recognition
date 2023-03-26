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
from tqdm import tqdm

import model_builder


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
        self.layers = model_builder.define_layers(name)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        # Run on GPU if available.
        self.to(get_processing_device())

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
        Optimize the neural network to fit the MNIST training data.
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
    for raw_image, raw_label in dataloader:
        image, label = to_tensor(raw_image), to_tensor(raw_label)
        # If properly predicted, count it as correct.
        correct += (label == model.predict(image)).sum()
    # Calculate the overall accuracy.
    return correct / (len(dataloader) * batch) * 100


def prepare_data(data):
    """
    Convert the raw pixel data strings for use with PyTorch.
    :param data: The data frame loaded from the CSV with emotions and pixel data.
    :return: Images and labels ready for PyTorch.
    """
    images = numpy.zeros(shape=(len(data), 48, 48))
    labels = numpy.array(list(map(int, data['emotion'])))
    # Break apart the string and resize it as a 48x48 image.
    for i, row in enumerate(data.index):
        image = numpy.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = numpy.reshape(image, (48, 48))
        images[i] = image
    # Ensure single channel added for proper inputs.
    images = images.reshape((images.shape[0], 48, 48, 1))
    # Scale all color values between 0 and 1.
    images = images.astype('float32') / 255
    return images, labels


def save(name: str, mode: str, model, best_model, epoch: int, best_accuracy: float, loss: float):
    torch.save({
        'Best': best_model,
        'Training': model.state_dict(),
        'Optimizer': model.optimizer.state_dict(),
        'Epoch': epoch,
        'Best Accuracy': best_accuracy,
        'Loss': loss
    }, f"{os.getcwd()}/Models/{name}-{mode}.pt")


def write_parameters(name: str, mode: str, best_accuracy: float, train_accuracy: float, inference_time: float, trainable_parameters: int, best_epoch: int):
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
        for mode in ["Normal", "Hybrid", "Gradual"]:
            model = NeuralNetwork(name)
            best_model = model.state_dict()
            # Check if an existing model of the same name exists.
            if os.path.exists(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.pt")):
                try:
                    saved = torch.load(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.pt"))
                    epoch = saved['Epoch']
                    best_accuracy = saved['Best Accuracy']
                    # If already done training this joint, skip to the next.
                    if epoch >= epochs:
                        print(f"{name} | {mode} | Accuracy = {best_accuracy}%")
                        continue
                    best_model = saved['Best']
                    model.load_state_dict(saved['Training'])
                    model.optimizer.load_state_dict(saved['Optimizer'])
                    loss = saved['Loss']
                    print(f"Continuing training for {name} on mode {mode} from epoch {epoch} with batch size {batch} for {epochs} epochs.")
                except:
                    print("Unable to load training data, exiting.")
                    return
            else:
                loss = -1
                epoch = 1
                print(f"Starting training for {name} on mode {mode} with batch size {batch} for {epochs} epochs.")
            # Ensure folder to save models exists.
            if not os.path.exists(os.path.join(os.getcwd(), "Models")):
                os.mkdir(os.path.join(os.getcwd(), "Models"))
            start = time.time_ns()
            accuracy = test(model, batch, testing)
            end = time.time_ns()
            inference_time = ((end - start) / testing_total) / 1e+6
            # If new training, write initial files.
            if epoch == 1:
                best_accuracy = accuracy
                f = open(os.path.join(os.getcwd(), "Models", f"{name}-{mode}.csv"), "w")
                f.write("Epoch,Loss,Accuracy,Best Accuracy")
                f.close()
            train_accuracy = test(model, batch, training)
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            write_parameters(name, mode, best_accuracy, train_accuracy, inference_time, trainable_parameters, 0)
            save(name, mode, model, best_model, epoch, best_accuracy, loss)
            # Train for set epochs.
            while True:
                if epoch > epochs:
                    print(f"{name} | {mode} | Accuracy = {best_accuracy}%")
                    break
                loss_message = "Loss = " + (f"{loss:.4}" if epoch > 1 else "N/A")
                msg = f"{name} | {mode} | Epoch {epoch}/{epochs} | {loss_message} | Accuracy = {accuracy:.4}% | Best = {best_accuracy:.4}%"
                # Reset loss every epoch.
                loss = 0
                # Switch to training mode.
                model.train()
                if mode == "Normal":
                    training_data.set_transform(-1)
                elif mode == "Hybrid":
                    training_data.set_transform(1 if epoch >= epochs / 2 else -1)
                else:
                    training_data.set_transform((epoch - 1) / float(epochs - 1))
                training = DataLoader(training_data, batch_size=batch, shuffle=True)
                for raw_image, raw_label in tqdm(training, msg):
                    image, label = to_tensor(raw_image), to_tensor(raw_label)
                    loss += model.optimize(image, label)
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
