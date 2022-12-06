# COMP-4730 Project 2

- [Challenge](#challenge "Challenge")
- [Setup](#setup "Setup")
- [Usage](#usage "Usage")
- [Dataset](#dataset "Dataset")
- [Trained Models](#trained-models "Trained Models")
- [References](#references "References")

# Challenge

This codebase is for solving the [Kaggle Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge "Kaggle Challenges in Representation Learning: Facial Expression Recognition Challenge").

# Setup

1. Download the [dataset](#dataset) and optionally our [trained models](#trained-models "Trained Models").
2. Install [Python](https://www.python.org "Python").
   1. Python 3.7.10 was used for this project's creation, but any newer version should work.
3. Install required Python packages. If unfamiliar with Python, "pip" comes with standard Python installations, and you can run "pip *package*" to install a package.
   1. [PyTorch](https://pytorch.org "PyTorch") and TorchVision.
      1. It is recommended you visit [PyTorch's get started page](https://pytorch.org/get-started/locally "PyTorch Get Started") which will allow you to select to install CUDA support if you have an Nvidia GPU. This will give you a command you can copy and run to install PyTorch, TorchVision, and TorchAudio, but feel free to remove TorchAudio from the command as it is not needed. Check which version of CUDA your Nvidia GPU supports.
      2. When running the script, a message will be output to the console if it is using CUDA.
   2. The remaining packages can be installed via "pip *package*" in the console:
      1. [numpy](https://numpy.org "numpy")
      2. [pandas](https://pandas.pydata.org "pandas")
      3. [torchsummary](https://pypi.org/project/torchsummary "torchsummary")
      4. [torchviz](https://pypi.org/project/torchviz "torchviz")
      5. [tqdm](https://github.com/tqdm/tqdm "tqdm")
4. Install [Graphviz](https://graphviz.org "Graphviz") so the Python script can generate a graph of neural networks.

# Usage

1. Clone or download and extract this repository.
2. Follow [setup](#setup "Setup") steps.
3. "model_builder.py" is where you can build your model architecture.
   1. The input is always 1 channel 48x48, and the final output needs to be 7.
4. Run "main.py" with the name of the model, options are "simple", "expanded", or "resnet", with the following optional parameters:
   1. -e, --epoch - The number of epochs to train for. Defaults to 100.
   2. -b, --batch - Training and testing batch size. Defaults to 64.
   3. -t, --test - Load and test the model with the given name.
   4. -w, --wait - The number of epochs to wait before switching to augmented data if there are no network improvements. Defaults to 20.
5. Once done, a folder with the given name can be found in the "Models" folder which contains the following:
   1. "Model.pt" which contains the best weights and bias saved which can be loaded for inference.
   2. "Details.txt" which contains an overview of the model. 
   3. "Training.csv" which contains the loss and accuracy values for each training epoch.
   4. "Graph.png" which displays the network architecture.
   5. "Sample Unchanged.png" and "Sample Augmented.png" which show sample batches of the unchanged and augmented data.

# Dataset

1. Go [here on the Kaggle challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz "Kaggle Dataset").
2. Download "fer2013.tar.gz".
3. Extract "fer2013.tar.gz". You will likely need to do this twice unless the software you are using automatically recognizes it needs to be done twice, first to get to "fer2013.tar" and then extract it to get the folder "fer2013".
4. Inside the folder "fer2013", copy the file "fer2013.csv" into the root of your project directory.
5. Rename "fer2013.csv" to "Data.csv".

"Challenges in Representation Learning: A report on three machine learning contests."

I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li, X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio.

arXiv 2013.

# Trained Models

1. [Download our trained models here](https://uwin365-my.sharepoint.com/:u:/g/personal/rice118_uwindsor_ca/ER5EGzUZonJPvODMxZlTN0oBC3civzsIALIyRikks264nQ?e=i3z17o).
2. In the root of your project directory, extract "Models.zip" so you have a folder named "Models".

# References

- Based upon our codebase from the [first COMP-4730 project](https://github.com/StevenRice99/COMP-4730-Project-1 "COMP-4730 Project 1").
- Helper CSV parsing methods based upon [Kaggle submission by "DrCapa"](https://www.kaggle.com/code/drcapa/facial-expression-eda-cnn "Dr. Kappa Challenges in Representation Learning: Facial Expression Recognition Challenge").