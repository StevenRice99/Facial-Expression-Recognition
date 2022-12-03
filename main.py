import argparse
import os
import shutil
import time

import numpy
import pandas
import torch
import torchvision.utils
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
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


def data_image(dataloader, writer, title: str):
    """
    Generate a sample grid image of the dataset.
    :param dataloader: A dataloader.
    :param writer: The TensorBoard writer.
    :param title: The title to give the image.
    :return: Nothing.
    """
    writer.add_image(title, torchvision.utils.make_grid(iter(dataloader).__next__()[0]))


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


def accuracy_check(model, batch: int, dataloader, best_accuracy: float, name: str):
    accuracy = test(model, batch, dataloader)
    if accuracy > best_accuracy:
        # Save the model.
        if not os.path.exists(f"{os.getcwd()}/Models/{name}"):
            os.mkdir(f"{os.getcwd()}/Models/{name}")
        torch.save(model.state_dict(), f"{os.getcwd()}/Models/{name}/Network.pt")
    return accuracy


def main(name: str, epochs: int, batch: int, load: bool, augment: bool):
    """
    Main program execution.
    :param name: The name of the model to save files under.
    :param epochs: The number of training epochs.
    :param batch: The batch size.
    :param load: Whether to load an existing model or train a new one.
    :param augment: Whether training data should be augmented or not.
    :return: Nothing.
    """
    print(f"Face Expression Recognition Deep Learning\nRun 'tensorboard --logdir=runs' in '{os.getcwd()}' to see data.")
    if name == "Current":
        print("Cannot name the model 'Current' as that name is used for the current TensorBoard run and will get overwritten.")
        return
    print(f"Running on GPU with CUDA {torch.version.cuda}." if torch.cuda.is_available() else "Running on CPU.")
    # Setup datasets.
    print("Loading data...")
    if not os.path.exists(f"{os.getcwd()}/Data.csv"):
        print("Data.csv missing, visit https://github.com/StevenRice99/COMP-4730-Project-2#download-dataset for instructions.")
        return
    df = pandas.read_csv(f"{os.getcwd()}/Data.csv")
    train_images, train_labels = prepare_data(df[df['Usage'] == 'Training'])
    test_images, test_labels = prepare_data(df[df['Usage'] != 'Training'])
    training_data = FaceDataset(train_images, train_labels, augment)
    testing_data = FaceDataset(test_images, test_labels)
    training_total = dataset_details("Training", train_labels)
    testing_total = dataset_details("Testing", test_labels)
    # Setup dataloaders.
    training = DataLoader(training_data, batch_size=batch, shuffle=True)
    testing = DataLoader(testing_data, batch_size=batch, shuffle=True)
    if load:
        # If a model does not exist to load decide to generate a new model instead.
        if not os.path.exists(f"{os.getcwd()}/Models/{name}/Network.pt"):
            print(f"Model '{name}' does not exist to load.")
            return
        try:
            print(f"Loading '{name}'...")
            model = NeuralNetwork()
            model.load_state_dict(torch.load(f"{os.getcwd()}/Models/{name}/Network.pt"))
            print(f"Loaded '{name}'.")
        except:
            print("Model to load has different structure than 'model_builder'.py, cannot load.")
            return
    else:
        model = NeuralNetwork()
    summary(model, input_size=(1, 48, 48))
    if not load:
        accuracies = []
        losses = []
        best_epoch = -2
        best_accuracy = 0
        # Ensure folder to save models exists.
        if augment:
            print("Training data has been augmented.")
        else:
            print("Training data has not been augmented.")
        if not os.path.exists(f"{os.getcwd()}/Models"):
            os.mkdir(f"{os.getcwd()}/Models")
        # Check if an existing model of the same name exists.
        if os.path.exists(f"{os.getcwd()}/Models/{name}/Network.pt"):
            print(f"Model '{name}' already exists, checking its accuracy for comparison...")
            try:
                existing = NeuralNetwork()
                existing.load_state_dict(torch.load(f"{os.getcwd()}/Models/{name}/Network.pt"))
                best_accuracy = test(existing, batch, testing)
                print(
                    f"Existing '{name}' has accuracy of {best_accuracy:.4}%, will only save if better achieved.")
            except:
                print(f"Existing '{name}' has different structure than 'model_builder'.py, cannot compare.")
                return
        else:
            print(f"No existing saved model with name '{name}' to compare against.")
        # Setup TensorBoard writer.
        writer = SummaryWriter("runs/Current")
        # Train if we are building a new model, this is skipped if a model was loaded.
        print(f"Training '{name}' for {epochs} epochs with batch size {batch}...")
        # Train for the set number of epochs
        accuracy = accuracy_check(model, batch, testing, best_accuracy, name)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = -1
        if epochs < 1:
            epochs = 1
        for epoch in range(epochs):
            loss_message = "Previous Loss = " + (f"{loss:.4}" if epoch > 0 else "N/A")
            msg = f"Epoch {epoch + 1}/{epochs} | {loss_message} | Previous Accuracy = {accuracy:.4}%"
            # Reset loss every epoch.
            loss = 0
            for raw_image, raw_label in tqdm(training, msg):
                image, label = to_tensor(raw_image), to_tensor(raw_label)
                loss += model.optimize(image, label)
            loss /= training_total
            losses.append(loss)
            accuracy = accuracy_check(model, batch, testing, best_accuracy, name)
            accuracies.append(accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
            writer.add_scalar('Training Loss', loss, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
        print(f"Accuracy of last epoch = {accuracy:.4}%")
        if best_epoch == -2:
            print(
                f"Training Complete, did not perform better than existing '{name}' with {best_accuracy:.4}% accuracy.")
            writer.close()
            shutil.rmtree(f"{os.getcwd()}/runs/Current")
            return
        best_epoch += 1
        if best_epoch == 0:
            print(f"Model '{name}' performed best untrained.")
        else:
            print(f"Training '{name}' complete, best result was on epoch {best_epoch}.")
    # Test the best model on the testing data.
    print("Testing trained model...")
    # Reload best model from training as the final model may not have been the best.
    if not load:
        model.load_state_dict(torch.load(f"{os.getcwd()}/Models/{name}/Network.pt"))
    train_accuracy = test(model, batch, training)
    start = time.time_ns()
    accuracy = test(model, batch, testing)
    end = time.time_ns()
    inference_time = ((end - start) / testing_total) / 1e+6
    print(f"Testing Accuracy = {accuracy}%\n"
          f"Training Accuracy = {train_accuracy}%\n"
          f"Average Inference Time: {inference_time} ms")
    # Nothing to save if we are just testing inference on a loaded model.
    if load:
        return
    # Generate sample images of the data.
    data_image(training, writer, 'Training Sample')
    data_image(testing, writer, 'Testing Sample')
    # Write data to TensorBoard.
    writer.add_text(f"Accuracy", f"{accuracy}%")
    writer.add_text(f"Average Inference Time", f"{inference_time} ms")
    # Create a graph of the model.
    img = to_tensor(iter(testing).__next__()[0])
    writer.add_graph(model, img)
    # Ensure the folder to save graph images exists.
    try:
        y = model.forward(img)
        make_dot(y.mean(), params=dict(model.named_parameters())).render(f"{os.getcwd()}/Models/{name}/Graph", format="png")
        # Makes redundant file, remove it.
        os.remove(f"{os.getcwd()}/Models/{name}/Graph")
    except:
        "Could not generate graph image, make sure you have 'Graphviz' installed."
    writer.close()
    # If there is an older, worse run data with the same name, overwrite it.
    if os.path.exists(f"{os.getcwd()}/runs/{name}"):
        shutil.rmtree(f"{os.getcwd()}/runs/{name}")
    os.rename(f"{os.getcwd()}/runs/Current", f"{os.getcwd()}/runs/{name}")
    # Save model parameters and accuracy.
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f = open(f"{os.getcwd()}/Models/{name}/Details.txt", "w")
    f.write(f"Testing Accuracy: {best_accuracy}\n"
            f"Training Accuracy: {train_accuracy}\n"
            f"Average Inference Time: {inference_time} ms\n"
            f"Trainable Parameters: {trainable_parameters}\n"
            f"Best Epoch: {best_epoch}\n"
            f"Total Epochs: {epochs}\n"
            f"Batch Size: {batch}\n"
            f"Augmented: {augment}")
    f.close()
    # Write training data.
    f = open(f"{os.getcwd()}/Models/{name}/Training.csv", "w")
    f.write("Epoch,Loss,Accuracy")
    for epoch in range(epochs):
        f.write(f"\n{epoch + 1},{losses[epoch]},{accuracies[epoch]}")
    f.close()


if __name__ == '__main__':
    desc = "Face Expression Recognition Deep Learning\n-----------------------------------------"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)
    parser.add_argument("-n", "--name", type=str, help="Name of the model.", default='Model')
    parser.add_argument("-e", "--epoch", type=int, help="Number of training epochs.", default=10)
    parser.add_argument("-b", "--batch", type=int, help="Training and testing batch size.", default=64)
    parser.add_argument("-l", "--load", help="Load the model with the given name and test it.", action="store_true")
    parser.add_argument("-s", "--standard", help="Keep training data standard and apply no transformations.", action="store_false")
    a = vars(parser.parse_args())
    main(a["name"], a["epoch"], a["batch"], a["load"], a["standard"])
