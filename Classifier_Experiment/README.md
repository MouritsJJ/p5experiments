# Follow the instructions to run this experiment

# Create the Used Data Sets

In this experiment a new split of the data set is needed. First a 50/50 split of the original data set, where one part is for the classifier to train on and the second part is for the generator to train on.

`python split_set.py`

Afterwards a 80/20 split is made on the classifiers data set as it needs a validation set when training.

`python create_validation_set.py`

# Train and Run the Classifier and Generator

Navigate to `generator/train` and run

`python main.py`

This should train a generator that saves a model every 10th epoch. You should inspect the results and find the best model and place it in `generator/evaluate`. The model can be found in `generator/train/data/training_iteration_1`. Then navigate to `generator/evaluate` and run

`python main.py`

Now it is time to train the classifier and use it on the generated images. Navigate to `classifier/train` and run

`python main.py`

Finally it is time to classify the generated images. Navigate to `classifier/evaluate` and run

`python main.py`

You are now done and the terminal should print the results.

# Hyper Parameters

## Generator

Bias = False\
Batch size = 128\
image size = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
beta1 = (0.5, 0.999)\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 100

## Classifier

Bias = False\
Batch size = 64\
image size = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
beta1 = (0.5, 0.999)\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 20

# Models

## Generator Train & Evaluate

Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

## Classifier Train & Evaluate

Classifier: 128x128x3, 64x64x64, 32x32x128, 16x16x256, 8x8x512, 1x4096, 8

# Runs
As the e
xperiment focuses on the quality on the generated images of previous experiments, only a single model is trained and examined.\
It has no changes to the model or hyper parameters. 

## gen_run

learning rate = 0.001