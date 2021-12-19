"""
Code heavily inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (Accessed 19/12-2021)
with only small adjustments
"""
dataroot = '../../car_dataset_sorted'

stats = (0.5,0.5,0.5), (0.5,0.5,0.5)

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 120

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 40

# Learning rate for optimizers
lr = 0.0075

# Beta1 hyperparam for Adam optimizers
beta_params = (0.5, 0.999)

# Number of gpus
ngpu = 1

# Training times per epoch for discriminator
disc_training_times = 1

# Training times per epoch for generator
gen_training_times = 1

# Training iteration
training_iteration = 1

# Number of labels
n_labels = 8

# Embedded dimension
embedded_dimension = 100