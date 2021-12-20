"""
Code heavily inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html (Accessed 19/12-2021)
with only small adjustments
"""
dataroot = './car_dataset_sorted_classifier'

validation_dataroot = './car_dataset_sorted_classifier_validation'

project_name = 'sp9classifier'

entity_name = 'p5_synthetic_gan'

model_name = 'classifier_model'

stats = (0.5,0.5,0.5), (0.5,0.5,0.5)

# Patience for early stopping
patience = 7

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 64

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
num_epochs = 20

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta_params = (0.5, 0.999)

# Number of gpus
ngpu = 1

# Training iteration
training_iteration = 1

# Number of labels
n_labels = 8

# Embedded dimension
embedded_dimension = 100
