# -*- coding: utf-8 -*-


import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np
import re
import hashlib
import piexif
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1


def move_fungi(old_dir, new_dir, validation_percentage, testing_percentage):

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    else:
        shutil.rmtree(new_dir)
        os.mkdir(new_dir)

    for i in ['train', 'val', 'test']:
        os.mkdir(os.path.join(new_dir, i))

    for i in tqdm(os.listdir(old_dir)):

        old_name = os.path.join(old_dir, i)
        if n_files(old_name) > 20:
            for j in ['train', 'val', 'test']:
                new_name = os.path.join(new_dir, j, i)


                if not os.path.exists(new_name):
                    os.mkdir(new_name)

            for j in os.listdir(old_name):
                old_image_name = os.path.join(old_name, j)

                # from TF poets

                hash_name = re.sub(r'_nohash_.*$', '', old_image_name)
                hash_name_hashed = hashlib.sha1(str.encode(hash_name)).hexdigest()
                percentage_hash = ((int(hash_name_hashed, 16) %
                                   (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                   (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                
                namy = ".".join([j.split(".")[0], j.split(".")[-1].lower()])
                if percentage_hash < validation_percentage:
                    new_image_name = os.path.join(new_dir,
                                                  'val',
                                                  i,
                                                  namy)
                elif percentage_hash < (testing_percentage +
                                        validation_percentage):
                    new_image_name = os.path.join(new_dir,
                                                  'test',
                                                  i,
                                                  namy)
                else:
                    new_image_name = os.path.join(new_dir,
                                                  'train',
                                                  i,
                                                  namy)

                shutil.copy(old_image_name, new_image_name)
                piexif.remove(new_image_name)


def n_files(direc):
    return len([j for j in os.listdir(direc)])



if __name__ == "__main__":
    m, s = get_mean_n_std("../with_fungi_dataset/images/")
    # np.save("./data_info.npy", np.array([m, s]))
    move_fungi("./images_use", "./images_app", 20, 20)
