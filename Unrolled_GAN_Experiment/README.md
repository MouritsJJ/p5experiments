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

# Base Model
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

# Runs
## Runs using the Base Model
### base_lr001_k1
Learning rate = 0.01\
Unrolled steps = 1

### base_lr001_k2
Learning rate = 0.01\
Unrolled steps = 2

### Base Model - lr0001_k1
Learning rate = 0.001\
Unrolled steps = 1

### base_lr0001_k2
Learning rate = 0.001\
Unrolled steps = 2

### base_lr0001_k3
Learning rate = 0.001\
Unrolled steps = 3

### base_lr0001_k5
Learning rate = 0.001\
Unrolled steps = 5

### base_lr00001_k1
Learning rate = 0.0001\
Unrolled steps = 1

### Model 13 - base_lr00001_k2
Learning rate = 0.0001\
Unrolled steps = 2

## Runs using the add_gen_128x16 model from the cDCGAN Layers Experiment
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x16, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_128x16_lr001_k2
Learning rate = 0.01\
Unrolled steps = 2

### add_gen_128x16_lr0001_k2
Learning rate = 0.001\
Unrolled steps = 2

### add_gen_128x16_lr0001_k3
Learning rate = 0.001\
Unrolled steps = 3

### add_gen_128x16_lr0001_k5
Learning rate = 0.001\
Unrolled steps = 5

### add_gen_128x16_lr00001_k2
Learning rate = 0.0001\
Unrolled steps = 2

## Runs using the model from the Initial cDCGAN Experiment
Gen: 4x513 → 8x256 → 16x128 → 64x64 → 128x3\
Dis: 128x4 → 64x64 → 16x128 → 4x256 → 1x1

### ICbase_lr001_k2
Learning rate = 0.01\
Unrolled steps = 2

### ICbase_lr0001_k2
Learning rate = 0.001\
Unrolled steps = 2

### ICbase_lr0001_k3
Learning rate = 0.001\
Unrolled steps = 3

### ICbase_lr0001_k5
Learning rate = 0.001\
Unrolled steps = 5

### ICbase_lr00001_k2
Learning rate = 0.0001\
Unrolled steps = 2

## Runs using Model 5 - rem_disc_32x64 from cDCGAN Layers Experiment
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 8x256, 4x512, 1x1

### model_5_lr0001_k1
Learning rate = 0.001\
Unrolled steps = 1

### model_5_lr0001_k2
Learning rate = 0.001\
Unrolled steps = 2

### model_5_lr0001_k3
Learning rate = 0.001\
Unrolled steps = 3

### model_5_lr0001_k5
Learning rate = 0.001\
Unrolled steps = 5