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
number of labels = 22\
embedded dimension = 100\
seed = 999\
epochs = 40

# Base Model
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x23

# Runs

## Runs using base model

### BCEWL LR 0-0002 B1 0-1
Learning rate = 0.0002
Beta1 = 0.1

### BCEWL LR 0-0005 B1 0-1
Learning rate = 0.0005
Beta1 = 0.1

### BCEWL LR 0-0002 B1 0-3
Learning rate = 0.0002
Beta1 = 0.3

### BCEWL LR 0-0005 B1 0-3
Learning rate = 0.0005
Beta1 = 0.3

### BCEWL LR 0-0002 B1 0-7
Learning rate = 0.0002
Beta1 = 0.7

### BCEWL LR 0-0005 B1 0-7
Learning rate = 0.0005
Beta1 = 0.7

### BCEWL LR 0-0002 B1 0-9
Learning rate = 0.0002
Beta1 = 0.9

### BCEWL LR 0-0005 B1 0-9
Learning rate = 0.0005
Beta1 = 0.9
