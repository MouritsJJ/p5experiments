from keras.datasets import mnist
from matplotlib import image
import matplotlib.pyplot as plt
from model import GAN
from PIL import Image
from pathlib import Path
import numpy as np

image_px = 90
image_list = []
labels_list = []
for filename in Path(r'../512_cars/cars_from_side').glob('*'):
    im = Image.open(filename)
    im = im.resize((image_px, image_px))
    im = im.convert('L')
    npim = np.array(im)
    image_list.append(npim)
    labels_list.append("car")

image_list = np.array(image_list)
labels_list = np.array(labels_list)
print("image list", image_list.shape)
print("labels list", labels_list.shape)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print("y_train.shape",y_train.shape)
# print("x_train.shape",x_train.shape)
# print("y_test.shape",y_test.shape)
# print("x_test.shape",x_test.shape)

labels = ["car"]

model = GAN(labels, learning_rate=0.001, decay_rate=0.00001, epochs=500, image_size=image_px,
    hidden_layer_size_g=128, hidden_layer_size_d=128, input_layer_size_g=100, batch_size=64,
    create_gif=False)
J_Ds, J_Gs = model.train(image_list, labels_list)

# plt.plot([i for i in range(len(J_Ds))], J_Ds)
# plt.plot([i for i in range(len(J_Gs))], J_Gs)

# plt.xlabel("# training steps")
# plt.ylabel("training cost")
# plt.legend(['Discriminator', 'Adversarial'])
# plt.show()
