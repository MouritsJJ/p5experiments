# Hyper Parameters
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

# Base Model
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

# Runs
## Runs removing layers from the discriminator
### Model 4 - rem_disc_4x512
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 8x256, 1x1

### rem_disc_8x256
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 4x512, 1x1

### Model 5 - rem_disc_32x64
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 8x256, 4x512, 1x1

## Runs adding layers to the generator
### add_gen_48x96
Gen: 4x1025, 8x512, 16x256, 32x128, 48x96, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_6x768
Gen: 4x1025, 6x768 , 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_128x16
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x16, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_6x768_48x96
Gen: 4x1025, 6x768, 8x512, 16x256, 32x128, 48x96, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_48x96_128x16
Gen: 4x1025, 8x512, 16x256, 32x128, 48x96, 64x64, 128x32, 128x16, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_6x768_128x16
Gen: 4x1025, 6x768 , 8x512, 16x256, 32x128, 64x64, 128x32, 128x16, 128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1

### add_gen_6x768_48x96_128x16
Gen: 4x1025, 6x768 , 8x512, 16x256, 32x128, 48x96, 64x64, 128x32, 128x16,  128x3\
Disc: 128x4, 32x64, 8x256, 4x512, 1x1
