# Models
## Model 1
Generator: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32, 128x3\
Discriminator: 128x4, 32x64, 8x256, 4x512, 1x1

## Model 4
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 32x64, 8x256, 1x1

## Model 5
Gen: 4x1025, 8x512, 16x256, 32x128, 64x64, 128x32,  128x3\
Disc: 128x4, 8x256, 4x512, 1x1

# Hyperparameters
Bias = False\
Batch size = 128\
image size (input) = 128 px\
nz = 120\
ngf (feature map, gen) = 64\
ndf (feature map, disc) = 64\
learning rate = 0.001\
beta1 = (0.5, 0.999)\
disc training per epoch = 1\
gen training per epoch = 1\
number of labels = 8\
embedded dimension = 100\
seed = 999\
epochs = 20

# Properties

## Original data set
Angle   | Images    | Per cent
|-------|:---------:|-------:|
East    | 1110      | 2.726
North   | 4483      | 11.009
NE      | 1508      | 3.703
NW      | 10362     | 25.446
South   | 5119      | 12.571
SE      | 3207      | 7.876
SW      | 9232      | 22.671
West    | 5700      | 13.998
Total   | 40721     | 100

## New data set limited with same per cent
Angle   | Images    | Per cent
|-------|:---------:|-------:|
East    | 2544      | 2.726
North   | 10274     | 11.009
NE      | 3456      | 3.703
NW      | 23748     | 25.447
South   | 11732     | 12.571
SE      | 7350      | 7.876
SW      | 21158     | 22.671
West    | 13063     | 13.997
Total   | 93325     | 100

## New data set 
Angle   | Images    | Per cent
|-------|:---------:|-------:|
East    | 28341     | 14.566
North   | 29649     | 15.238
NE      | 18914     | 9.721
NW      | 28932     | 14.869
South   | 27507     | 14.137
SE      | 22964     | 11.802
SW      | 21159     | 10.874
West    | 17110     | 8.793
Total   | 194576    | 100

# Runs
## ExtraN
Runs having only `extraN` in the name uses the largest data set.

## ExtraN_sameD
Runs having the extension `extraN_sameD` uses the data set with the same distribution of the angles but more images than the original data set.

## sameN_sameD
Runs having the extension `sameN_sameD` uses the new data set but with the same amount of images and the same distribution of angles as the original data set. 