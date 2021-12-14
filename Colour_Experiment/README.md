# Hyper Parameters

## BCEWL, cDCGAN
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
epochs = 20

## DCGAN
Bias = False\
Batch size = 32\
image size = 64 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
beta1 = (0.5, 0.999)\
disc training per epoch = 1\
gen training per epoch = 1\
seed = 999\
epochs = 20

## Unroll
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
epochs = 20\
unroll steps = 2

# Models

## Generator
BCEWL, cDCGAN, Unrolled: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
DCGAN: 4x512, 8x256, 16x128, 32x64, 32x3

## Discriminator
BCEWL: 128x4, 32x64, 8x256, 4x512, 1x23\
cDCGAN: 128x4, 32x64, 8x256, 4x512, 1x1\
DCGAN: 128x4, 32x64, 8x256, 4x512, 1x1\
Unroll: 128x4, 8x256, 4x512, 1x1

# Runs

## BCEWL

Learning rate = 0.001\
number of labels = 22\
embedded dimension = 100

## cDCGAN

Learning rate = 0.001\
number of labels = 22\
embedded dimension = 100

## DCGAN

Learning rate = 0.0002

## Unroll

Learning rate = 0.001\
number of labels = 22\
embedded dimension = 100\
unroll steps = 2