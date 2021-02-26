import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

import matplotlib.pyplot as plt

########################
########################

from pathlib import Path
ROOT_DIR_PATH = str(Path(__file__).parent.parents[0].resolve(strict=True))

########################
########################


data_path = '/data/Akeaveny/Datasets/part-affordance_combined/real/combined_tools1_'
new_data_path = '/data/Akeaveny/Datasets/domain_adaptation/UMD/Real/'

splits = [
    'train/',
    'val/',
    'test/',
]

image_exts = [
            '_rgb.jpg',
            '_depth.png',
            '_label.png'
]

########################
########################
for split in splits:
    offset = 0
    for image_ext in image_exts:
        file_path = data_path + split + '*' + image_ext
        files = np.array(sorted(glob.glob(file_path)))
        print("\nLoaded files: ", len(files))
        print("File path: ", file_path)

        if image_ext == '.png':
            offset += len(files)

        ###################
        ###################

        for idx, file in enumerate(files):
            old_file_name = file
            new_file_name = new_data_path + split

            count = 1000000 + offset + idx
            image_num = str(count)[1:]
            # print(f'\nImage num {image_num}')

            if image_ext == '_rgb.jpg':
                move_file_name = new_file_name + 'rgb/' + np.str(image_num) + '.jpg'
                if idx == 0 and split == 'train/':
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '_depth.png':
                move_file_name = new_file_name + 'depth/' + np.str(image_num) + '_depth.png'
                if idx == 0 and split == 'train/':
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '_label.png':
                move_file_name = new_file_name + 'masks/' + np.str(image_num) + '_label.png'
                if idx == 0 and split == 'train/':
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            else:
                print("*** IMAGE EXT DOESN'T EXIST ***")
                exit(1)