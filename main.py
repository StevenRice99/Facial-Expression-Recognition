import argparse
import os
import time

import numpy
import pandas
import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from torchviz import make_dot
from tqdm import tqdm

import model_builder


class FaceDataset(Dataset):
    """
    The datasets built from the CSV data.
    """

    def __init__(self, images, labels, augment: bool = False):
        """
        Create the dataset.
        :param images: The images parsed from the CSV.
        :param labels: The labels parsed from the CSV.
        :param augment: True to augment data augmented, false otherwise.
        """
        self.images = torch.tensor(images, dtype=torch.float32)
        # Rearrange axis, so they are in proper input order and appear visually upright.
        self.images = torch.swapaxes(self.images, 1, 3)
        self.images = torch.swapaxes(self.images, 2, 3)
        self.labels = torch.tensor(labels, dtype=torch.int)
        if augment:
            self.transform = transforms.Compose([
                # Randomly flip the image.
                transforms.RandomHorizontalFlip(),
                # Randomly adjust pixel values.
                transforms.ColorJitter(brightness=0.5, contrast=0.3, hue=0.3, saturation=0.3),
                # Randomly rotate the training data to add more variety.
                transforms.RandomRotation(45),
                # Randomly adjust image perspective.
                transforms.RandomPerspective(),
                # Randomly zoom in on training data to add more variety.
                transforms.RandomResizedCrop(48, (0.7, 1))
            ])
        else:
            self.transform = None

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

    def __init__(self):
        """
        Set up the neural network loading in parameters defined in 'model_builder.py'.
        """
        super().__init__()
        # Load in defined parameters.
        self.layers = model_builder.define_layers()
        self.loss = model_builder.define_loss()
        self.optimizer = model_builder.define_optimizer(self)
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


def data_image(dataloader, name: str, title: str):
    """
    Generate a sample grid image of the dataset.
    :param dataloader: A dataloader.
    :param name: The name of the model.
    :param title: The title to give the image.
    :return: Nothing.
    """
    torchvision.utils.save_image(torchvision.utils.make_grid(iter(dataloader).__next__()[0]), f"{os.getcwd()}/Models/{name}/{title}.png")


def test(model, batch: int, dataloader):
    """
    Test a neural network.
    :param model: The neural network.
    :param batch: The batch size.
    :param dataloader: The dataloader to test.
    :return: The model's accuracy.
    """
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

def save(name: str, model, best_model, epoch: int, no_change: int, best_accuracy: float, loss: float):
    torch.save({
        'Best': best_model,
        'Training': model.state_dict(),
        'Optimizer': model.optimizer.state_dict(),
        'Epoch': epoch,
        'No Change': no_change,
        'Best Accuracy': best_accuracy,
        'Loss': loss
    }, f"{os.getcwd()}/Models/{name}/Model.pt")


def write_parameters(name: str, best_accuracy: float, train_accuracy: float, inference_time: float, trainable_parameters: int, best_epoch: int, batch: int, augment: bool):
    parameters = open(f"{os.getcwd()}/Models/{name}/Details.txt", "w")
    parameters.write(f"Testing Accuracy: {best_accuracy}\n"
                     f"Training Accuracy: {train_accuracy}\n"
                     f"Average Inference Time: {inference_time} ms\n"
                     f"Trainable Parameters: {trainable_parameters}\n"
                     f"Best Epoch: {best_epoch}\n"
                     f"Batch Size: {batch}\n"
                     f"Augmented: {augment}")
    parameters.close()

def main(name: str, batch: int, load: bool, augment: bool):
    """
    Main program execution.
    :param name: The name of the model to save files under.
    :param batch: The batch size.
    :param load: Whether to load an existing model or train a new one.
    :param augment: Whether training data should be augmented or not.
    :return: Nothing.
    """
    print(f"Face Expression Recognition Deep Learning")
    print(f"Running on GPU with CUDA {torch.version.cuda}." if torch.cuda.is_available() else "Running on CPU.")
    if not os.path.exists(f"{os.getcwd()}/Data.csv"):
        print("Data.csv missing, visit https://github.com/StevenRice99/COMP-4730-Project-2#download-dataset for instructions.")
        return
    # Setup datasets.
    print("Loading data...")
    df = pandas.read_csv(f"{os.getcwd()}/Data.csv")
    train_images, train_labels = prepare_data(df[df['Usage'] == 'Training'])
    test_images, test_labels = prepare_data(df[df['Usage'] != 'Training'])
    training_data = FaceDataset(train_images, train_labels, augment)
    testing_data = FaceDataset(test_images, test_labels)
    if augment:
        normal_training_data = FaceDataset(train_images, train_labels)
    else:
        normal_training_data = training_data
    training_total = dataset_details("Training", train_labels)
    testing_total = dataset_details("Testing", test_labels)
    # Setup dataloaders.
    training = DataLoader(training_data, batch_size=batch, shuffle=True)
    testing = DataLoader(testing_data, batch_size=batch, shuffle=True)
    normal_training = DataLoader(normal_training_data, batch_size=batch, shuffle=True)
    if load:
        # If a model does not exist to load decide to generate a new model instead.
        if not os.path.exists(f"{os.getcwd()}/Models/{name}/Model.pt"):
            print(f"Model '{name}' does not exist to load.")
            return
        try:
            model = NeuralNetwork()
            saved = torch.load(f"{os.getcwd()}/Models/{name}/Model.pt")
            model.load_state_dict(saved['Best'])
        except:
            print("Model to load has different structure than 'model_builder'.py, cannot load.")
            return
        train_accuracy = test(model, batch, normal_training)
        start = time.time_ns()
        accuracy = test(model, batch, testing)
        end = time.time_ns()
        inference_time = ((end - start) / testing_total) / 1e+6
        print(f"Testing Accuracy = {accuracy}%\n"
              f"Training Accuracy = {train_accuracy}%\n"
              f"Average Inference Time: {inference_time} ms")
        return
    model = NeuralNetwork()
    best_model = model.state_dict()
    summary(model, input_size=(1, 48, 48))
    # Check if an existing model of the same name exists.
    if os.path.exists(f"{os.getcwd()}/Models/{name}/Model.pt"):
        try:
            print(f"Model '{name}' already exists, attempting to load to continue training...")
            saved = torch.load(f"{os.getcwd()}/Models/{name}/Model.pt")
            best_model = saved['Best']
            model.load_state_dict(saved['Training'])
            model.optimizer.load_state_dict(saved['Optimizer'])
            epoch = saved['Epoch']
            no_change = saved['No Change']
            best_accuracy = saved['Best Accuracy']
            loss = saved['Loss']
            print(f"Continuing training for '{name}' from epoch {epoch}.")
        except:
            print("Unable to load training data, exiting.")
    else:
        loss = -1
        epoch = 1
        no_change = 0
    if not os.path.exists(f"{os.getcwd()}/Models/{name}"):
        os.mkdir(f"{os.getcwd()}/Models/{name}")
    if augment:
        print("Training data has been augmented.")
    else:
        print("Training data has not been augmented.")
    if batch < 1:
        batch = 1
    print(f"Running training with batch size {batch}. Stop program to end training.")
    # Ensure folder to save models exists.
    if not os.path.exists(f"{os.getcwd()}/Models"):
        os.mkdir(f"{os.getcwd()}/Models")
    start = time.time_ns()
    accuracy = test(model, batch, testing)
    end = time.time_ns()
    inference_time = ((end - start) / testing_total) / 1e+6
    if epoch == 1:
        best_accuracy = accuracy
        f = open(f"{os.getcwd()}/Models/{name}/Training.csv", "w")
        f.write("Epoch,Loss,Accuracy,Best Accuracy")
        f.close()
        # Generate sample images of the data.
        data_image(training, name, 'Training Sample')
        data_image(testing, name, 'Testing Sample')
        # Create a graph of the model.
        try:
            y = model.forward(to_tensor(iter(testing).__next__()[0]))
            make_dot(y.mean(), params=dict(model.named_parameters())).render(f"{os.getcwd()}/Models/{name}/Graph",
                                                                             format="png")
            # Makes redundant file, remove it.
            os.remove(f"{os.getcwd()}/Models/{name}/Graph")
        except:
            "Could not generate graph image, make sure you have 'Graphviz' installed."
    train_accuracy = test(model, batch, normal_training)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_parameters(name, best_accuracy, train_accuracy, inference_time, trainable_parameters, 0, batch, augment)
    save(name, model, best_model, epoch, no_change, best_accuracy, loss)
    # Train until program is stopped.
    while True:
        loss_message = "Previous Loss = " + (f"{loss:.4}" if epoch > 1 else "N/A")
        improvement = "Improvement " if no_change == 0 else f"{no_change} Epochs No Improvement "
        msg = f"Epoch {epoch} | {loss_message} | Previous = {accuracy:.4}% | Best = {best_accuracy:.4}% | {improvement}"
        # Reset loss every epoch.
        loss = 0
        for raw_image, raw_label in tqdm(training, msg):
            image, label = to_tensor(raw_image), to_tensor(raw_label)
            loss += model.optimize(image, label)
        loss /= training_total
        start = time.time_ns()
        accuracy = test(model, batch, testing)
        end = time.time_ns()
        if accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = accuracy
            inference_time = ((end - start) / testing_total) / 1e+6
            no_change = 0
            train_accuracy = test(model, batch, normal_training)
            write_parameters(name, best_accuracy, train_accuracy, inference_time, trainable_parameters, epoch, batch, augment)
        else:
            no_change += 1
        f = open(f"{os.getcwd()}/Models/{name}/Training.csv", "a")
        f.write(f"\n{epoch},{loss},{accuracy},{best_accuracy}")
        f.close()
        epoch += 1
        save(name, model, best_model, epoch, no_change, best_accuracy, loss)


if __name__ == '__main__':
    desc = "Face Expression Recognition Deep Learning\n-----------------------------------------"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument("-n", "--name", type=str, help="Name of the model.", default='Model')
    parser.add_argument("-b", "--batch", type=int, help="Training and testing batch size.", default=64)
    parser.add_argument("-t", "--test", help="Load and test the model with the given name.", action="store_true")
    parser.add_argument("-a", "--augment", help="Augment training data.", action="store_true")
    a = vars(parser.parse_args())
    main(a["name"], a["batch"], a["test"], a["augment"])
