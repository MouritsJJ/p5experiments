# Hyper Parameters
Bias = False\
Batch size = 128\
image size = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
RMSProp momentum = 0.00005\
disc training per epoch = 1\
gen training per epoch = 1\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 20
learning rate = 0.00005\
n_critic = 1\
clipping range = 0.01

# Base Model
Gen: 4x513 → 8x256 → 16x128 → 64x64 → 128x3\
Dis: 128x4 → 64x64 → 16x128 → 4x256 → 1x1

# Runs
## Runs training the critic 3 times per iteration
### nc3_cr0005
n_critic = 3\
clipping range = 0.005

### nc3_cr001
n_critic = 3\
clipping range = 0.01

### nc3_cr002
n_critic = 3\
clipping range = 0.02

### nc3_cr003
n_critic = 3\
clipping range = 0.03

### nc3_cr004
n_critic = 3\
clipping range = 0.04

### nc3_cr005
n_critic = 3\
clipping range = 0.05

### nc3_cr01
n_critic = 3\
clipping range = 0.1

## Runs training the critic 5 times per iteration
### Model 14 - nc5_cr0005
n_critic = 5\
clipping range = 0.005

### nc5_cr001
n_critic = 5\
clipping range = 0.01

### nc5_cr002
n_critic = 5\
clipping range = 0.02

### nc5_cr003
n_critic = 5\
clipping range = 0.03

### nc5_cr004
n_critic = 5\
clipping range = 0.04

### nc5_cr005
n_critic = 5\
clipping range = 0.05

### nc5_cr01
n_critic = 5\
clipping range = 0.1

## Runs training the critic 7 times per iteration
### nc7_cr0005
n_critic = 7\
clipping range = 0.005

### nc7_cr001
n_critic = 7\
clipping range = 0.01

### nc7_cr002
n_critic = 7\
clipping range = 0.02

### nc7_cr003
n_critic = 7\
clipping range = 0.03

### nc7_cr004
n_critic = 7\
clipping range = 0.04

### nc7_cr005
n_critic = 7\
clipping range = 0.05

### nc7_cr01
n_critic = 7\
clipping range = 0.1

## Runs training the critic 9 times per iteration
### nc9_cr0005
n_critic = 9\
clipping range = 0.005

### nc9_cr001
n_critic = 9\
clipping range = 0.01

### nc9_cr002
n_critic = 9\
clipping range = 0.02

### nc9_cr003
n_critic = 9\
clipping range = 0.03

### nc9_cr004
n_critic = 9\
clipping range = 0.04

### nc9_cr005
n_critic = 9\
clipping range = 0.05

### nc9_cr01
n_critic = 9\
clipping range = 0.1

## Runs varying learning rate for a Wasserstein GAN
### nc3_cr001_lr000005
n_critic = 3\
clipping range = 0.01\
learning rate = 0.00005

### nc3_cr001_lr00001
n_critic = 3\
clipping range = 0.01\
learning rate = 0.0001

### nc3_cr001_lr0001
n_critic = 3\
clipping range = 0.01\
learning rate = 0.001

### nc3_cr001_lr001
n_critic = 3\
clipping range = 0.01\
learning rate = 0.01

### nc3_cr001_lr01
n_critic = 3\
clipping range = 0.01\
learning rate = 0.1

# Special Run - Model 15 - 10nc30
## Model
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3
Disc: 128x4, 64x64, 32x128, 16x256, 8x512, 4x1024, 1x1

## Parameters
epochs = 100\
learning rate = 0.00005\
RMSProp momentum = 0.00005\
clipping range = 0.005\
n_critic = 5

The model trains the critic 5 times per iteration as a standard rule. However, the first 10 epochs and each 20th epoch the critic will be trained 30 times per iteration.