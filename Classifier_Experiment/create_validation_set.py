from os import listdir, makedirs
from shutil import move
from random import shuffle
from math import floor

if __name__ == "__main__":
    from_folder_dirs = listdir('./car_dataset_sorted_classifier')
    validation_set_size = 0.2 # 20%

    for dir in from_folder_dirs:
        print(dir)
        makedirs(f'./car_dataset_sorted_classifier_validation/{dir}', exist_ok=True)

        images = listdir(f'./car_dataset_sorted_classifier/{dir}')
        shuffle(images)

        for i in range(len(images)):
            if i < floor(len(images) * validation_set_size):
                move(f'./car_dataset_sorted_classifier/{dir}/{images[i]}', f'./car_dataset_sorted_classifier_validation/{dir}/{images[i]}')
