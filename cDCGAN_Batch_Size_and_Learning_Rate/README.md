# Base Model
Bias = False\
Batch size = 32\
image size (input) = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
learning rate = 0.0002\
beta1 = (0.5, 0.999)\
disc training per epoch = 1\
gen training per epoch = 1\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 40

## Dimensions
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

# Runs
## Runs with a batch size of 128
### Model 9 - bs128_lr001
Learning rate: 0.01

### bs128_lr00025
Learning rate: 0.0025

### bs128_lr0005
Learning rate: 0.005

### bs128_lr00075
Learning rate: 0.0075

### Model 10 - bs128_lr0001
Learning rate: 0.001

### bs128_lr00001
Learning rate: 0.0001

### bs128_lr000001
Learning rate: 0.00001

### bs128_lr0000001
Learning rate: 0.000001

## Runs with a batch size of 256
### Model 11 - bs256_lr001
Learning rate: 0.01 

### Model 12 - bs256_lr0001
Learning rate: 0.001 

### bs256_lr00001
Learning rate: 0.0001

### bs256_lr000001
Learning rate: 0.00001 

### bs256_lr0000001
Learning rate: 0.000001 
