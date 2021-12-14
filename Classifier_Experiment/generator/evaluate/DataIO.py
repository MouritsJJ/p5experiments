import os
import csv
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch import save, load
from pathlib import Path

matplotlib.use('agg')

class DataIO:

    def __init__(self, folder_name, iteration):
        """
        Input:
            folder_name - name of the folder the images, figures and data are saved in
        """
        # new folder in current directory
        self.path = Path(f'./{folder_name}/training_iteration_{iteration}')  
        folder_name = Path(f'./{folder_name}')
        if not folder_name.is_dir():
            folder_name.mkdir()

        if self.path.is_dir():
            print(f'Path: {self.path} exists')
            exit()
        else:
            self.path.mkdir()

    def save_single_image(self, img, image_name, title = ''):
        plt.figure()
        plt.axis("off")
        plt.title(title)
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.savefig(f'{self.path}/{image_name}.png', bbox_inches='tight', pad_inches = 0, dpi=34)
        plt.close()

    def save_last_image(self, image_name, img):
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.savefig(f'{self.path}/{image_name}.png')
        plt.close()

    def create_loss_image(self, D_losses, G_losses):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()
        plt.savefig(f"{self.path}/graph.png")

    def save_cost(self, generator_cost, discriminator_cost, file_name):
            """
            Saves the data from generator_cost and discriminator_cost to file_name
            Row 0 is generator_cost and row 1 is discriminator_cost
            Input:
                generator_cost - list of generator costs through the training process
                discriminator_cost - list of discriminator costs through the training process
                file_name - name of the .csv file containing the data. 
            """

            # Deletes .csv file with file_name if it exists
            if os.path.exists(f'{pathlib.Path(__file__).parent.resolve()}/{self.path}/{file_name}.csv'):
                os.remove(f'{self.path}/{file_name}.csv')

            # Write to file_name .csv
            with open(f'{self.path}/{file_name}.csv', mode='w') as data:
                data_writer = csv.writer(data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                # Header
                data_writer.writerow(['generator_cost', 'discriminator_cost'])
                for i in range(len(generator_cost)):
                    data_writer.writerow([generator_cost[i], discriminator_cost[i]])
        
    def read_cost(self, file_name):
        """
        Reads the data from file_name and returns the two lists, generator_cost and discriminator_cost
        Input:
            file_name - name of the file to read from
        Output:
            generator_cost - list of generator costs through the training process
            discriminator_cost - list of discriminator costs through the training process
        """

        generator_cost, discriminator_cost = [], []

        # Read data
        with open(f'{self.path}/{file_name}.csv') as data_reader:
            # Skips the header
            next(data_reader)
            for row in data_reader:
                generator_cost.append(row[0])
                discriminator_cost.append(row[2])
        
        return generator_cost, discriminator_cost

    def save_classifier(self, classifier, classifier_op, file_name):
        save({
                'classifier': classifier.state_dict(),
                'classifier_op': classifier_op.state_dict(),
            }, f'{self.path}/{file_name}')

    def save_models(self, disc, gen, disc_op, gen_op, file_name):
        """
        Saves the model to file_name
        Input:
            disc - discriminator (nn.module)
            gen - generator (nn.module)
            disc_op - disciminator optimizer (torch)
            gen_op - generator optimizer (torch)
            file_name - file name to save to
        """

        save({
                'discriminator': disc.state_dict(),
                'generator': gen.state_dict(),
                'disc_op': disc_op.state_dict(),
                'gen_op': gen_op.state_dict()
            }, f'{self.path}/{file_name}')

    def load_classifier(self, classifier, classifier_op, model_path):
        models = load(model_path)
        classifier.load_state_dict(models['classifier'])
        classifier_op.load_state_dict(models['classifier_op'])

        classifier.eval()

    def load_models(self, disc, gen, disc_op, gen_op, file_name):
        """
        Loads the model in file_name
        Input:
            disc - discriminator (nn.module)
            gen - generator (nn.module)
            disc_op - disciminator optimizer (torch)
            gen_op - generator optimizer (torch)
            file_name - file name to read from
        """

        models = load(f'./{file_name}')

        disc.load_state_dict(models['discriminator'])
        gen.load_state_dict(models['generator'])
        disc_op.load_state_dict(models['disc_op'])
        gen_op.load_state_dict(models['gen_op'])

        disc.eval()
        gen.eval()
