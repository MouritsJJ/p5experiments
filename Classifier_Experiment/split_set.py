from os import listdir, makedirs
from shutil import copyfile
from random import shuffle
from math import floor

if __name__ == "__main__":
    from_folder_dirs = listdir('../car_dataset_sorted')
    split_distribution = 0.5 # 50%

    for dir in from_folder_dirs:
        print(dir)
        makedirs(f'./car_dataset_sorted_classifier/{dir}', exist_ok=True)
        makedirs(f'./car_dataset_sorted_generator_training/{dir}', exist_ok=True)

        images = listdir(f'./car_dataset_sorted/{dir}')
        shuffle(images)

        for i in range(len(images)):
            if i < floor(len(images)* split_distribution):
                copyfile(f'../car_dataset_sorted/{dir}/{images[i]}', f'./car_dataset_sorted_classifier/{dir}/{images[i]}')
            else:
                copyfile(f'../car_dataset_sorted/{dir}/{images[i]}', f'./car_dataset_sorted_generator_training/{dir}/{images[i]}')
