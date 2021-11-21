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

## Model Dimensions
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

# Runs
## Runs concerning learning rate
### lr-1
Learning rate = 1.0

### lr-01
Learning rate = 0.1

### lr-001
Learning rate = 0.01

### Model 6 - lr-0001
Learning rate = 0.001

### lr-00001
Learning rate = 0.0001

### lr-000001
Learning rate = 0.00001

## Runs concerning batch size
### bs-2
Batch size = 2

### bs-4
Batch size = 4

### bs-8
Batch size = 8

### bs-16
Batch size = 16

### bs-32
Batch size = 32 (Base Model)

### bs-64
Batch size = 64

### Model 7 - bs-128
Batch size = 128

### bs-256
Batch size = 256

### bs-512
Batch size = 512

## Runs concerning noise vector
### noise-100
nz = 100

### noise-110
nz = 110

### noise-120
nz = 120 (Base Model)

### Model 8 - noise-130
nz = 130

### noise-140
nz = 140

## Runs concerning beta1
### beta1-01
beta1 = 0.1

### beta1-03
beta1 = 0.3

### beta1-05
beta1 = 0.5 (Base Model)

### beta1-07
beta1 = 0.7

### beta1-09
beta1 = 0.9

## Runs concerning beta2
### beta2-09
beta2 = 0.9

### beta2-099
beta2 = 0.99

### beta2-0999
beta2 = 0.999 (Base Model)

### beta2-09999
beta2 = 0.9999

### beta2-099999
beta2 = 0.99999