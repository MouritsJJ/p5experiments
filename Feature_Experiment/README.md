# Hyper Parameters

Bias = False\
Batch size = 128\
image size = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
beta1 = (0.5, 0.999)\
disc training per epoch = 1\
gen training per epoch = 1\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 20

# Models

## BCE
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x23

## Base
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

# Runs
To create the models used for generating images open a terminal, navigate to the `base` folder and run `python main.py`. A model names base_model should be saved in the main folder for the experiment.\
Hereafter, navigate to the `BCE` folder and run `python main.py`. A model named `BCE_model` should be saved in the main folder for the experiment.

Running the extraction and transition experiments should then be performed by navigation to the respective folder and running the same `python main.py` command.

## Feature Extraction
A noise 0-vector with 120 elements is generated and each element is then varied between -10 and 10, while all other elements are kept at zero. Each variation of the noise vector is used as input to both models and the generated images is saved to be analyzed.

## Feature Transition
A noise vector with 120 elements is generated and each element is then varied between -10 and 10, while all other elements are kept at their initial value. Each variation of the noise vector is used as input to both models and the generated images is saved to be analyzed.