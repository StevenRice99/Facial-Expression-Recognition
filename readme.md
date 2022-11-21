# COMP-4730 Project 2

- [Challenge](#challenge "Challenge")
- [Setup](#setup "Setup")
  - [Download Dataset](#download-dataset "Download Dataset")
- [Usage](#usage "Usage")
- [Limitations](#limitations "Limitations")
- [References](#references "References")
- [Dataset Reference](#dataset-reference "Dataset Reference")

# Challenge

This codebase is for solving the [Kaggle Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge, "Kaggle Challenges in Representation Learning: Facial Expression Recognition Challenge").

# Setup

1. Install [Python](https://www.python.org "Python").
   1. Python 3.7.10 was used for this project's creation, but any newer version should work.
2. Install required Python packages. If unfamiliar with Python, "pip" comes with standard Python installations, and you can run "pip *package*" to install a package.
   1. [PyTorch](https://pytorch.org "PyTorch") and TorchVision.
      1. It is recommended you visit [PyTorch's get started page](https://pytorch.org/get-started/locally "PyTorch Get Started") which will allow you to select to install CUDA support if you have an Nvidia GPU. This will give you a command you can copy and run to install PyTorch, TorchVision, and TorchAudio, but feel free to remove TorchAudio from the command as it is not needed. Check which version of CUDA your Nvidia GPU supports.
      2. When running the script, a message will be output to the console if it is using CUDA.
   2. The remaining packages can be installed via "pip *package*" in the console:
      1. [shutil](https://docs.python.org/3/library/shutil.html "shutil")
      2. [tensorboard](https://pypi.org/project/tensorboard "tensorboard")
      3. [torchsummary](https://pypi.org/project/torchsummary "torchsummary")
      4. [torchviz](https://pypi.org/project/torchviz "torchviz")
      5. [tqdm](https://github.com/tqdm/tqdm "tqdm")
3. Install [Graphviz](https://graphviz.org "Graphviz") so the Python script can generate a graph of neural networks.

## Download Dataset

1. Go [here on the Kaggle challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz "Kaggle Dataset").
2. Download "fer2013.tar.gz".
3. Extract "fer2013.tar.gz". You will likely need to do this twice unless the software you are using automatically recognizes it needs to be done twice, first to get to "fer2013.tar" and then extract it to get the folder "fer2013".
4. Inside the folder "fer2013", copy the file "fer2013.csv" into the root of your project directory.
5. Rename "fer2013.csv" to "Data.csv".

# Usage

1. Clone or download and extract this repository.
2. Follow [setup](#setup "Setup") steps.
3. "model_builder.py" is where you can build your model architecture.
   1. The input is always 1 channel 48x48, and the final output needs to be 7.
4. In the cloned/downloaded directory, run: **tensorboard --logdir="runs"** or **python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]**
   1. This will allow you to open TensorBoard in your browser which by default is usually at [http://localhost:6006](http://localhost:6006 "Tensorboard").
5. Run "main.py" with optional run parameters.
   1. -n, --name - Name of the model. Defaults to "Model".
   2. -e, --epoch - Number of training epochs. Defaults to 10.
   3. -b, --batch - Training and testing batch size. Defaults to 64.
   4. -l, --load - Load the model with the given name and test it.
   5. -a, --attempts - The amount of times to try the training. Default to 1.
   6. --min - Minimum rescaling size of training data. Defaults to 0.
   7. --max - Maximum rescaling size of training data. Defaults to 0.
   8. -r, --rot - Maximum rotation of training data. Defaults to 0.
6. Once done, the model:
   1. Will be saved with the name as it given in the "Models" folder.
   2. Architecture will be saved as a graph in the "Graphs" folder.
   3. Parameters and accuracy will be saved with the same model name in the "ModelParameters" folder.
   4. TensorBoard data will be saved in the "runs" folder.
   - Note if this run produced a worse accuracy than a previous run which had the same name, it will not be saved to preserve the best model.

# Limitations

- Loading a model will fail if you modified the architecture in "model_builder.py" from when the model you are trying to load was made.

# References

- Based upon our codebase from the [first COMP-4730 project](https://github.com/StevenRice99/COMP-4730-Project-1 "COMP-4730 Project 1").
- Helper CSV parsing methods based upon [Kaggle submission by "DrCapa"](https://www.kaggle.com/code/drcapa/facial-expression-eda-cnn "Dr. Kappa Challenges in Representation Learning: Facial Expression Recognition Challenge").

# Dataset Reference

"Challenges in Representation Learning: A report on three machine learning contests."

I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li, X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio.

arXiv 2013.