# In[]
import os
import random
from random import randint

import cv2
import numpy as np
from pyjeasy.image_utils.edit import resize_img
from pyjeasy.image_utils.output import show_image
from tqdm import tqdm


def folder_list(folder):
    files = [os.path.join(folder, fn) for fn in os.listdir(folder)]
    images = [fn for fn in files if os.path.splitext(
        fn)[1].lower() in ('.jpg', '.jpeg', '.png')]
    return images


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [vconcat_resize_min(
        im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    im_list = hconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)
    return im_list


# In[]
folder1 = "/home/jitesh/3d/data/images_for_ndds_bg/val2017"
image_names = folder_list(folder1)
images = []

for name in tqdm(image_names, desc="Loading raw images"):
    # open all images and find their sizes
    images.append(cv2.imread(name))
    
# In[]
output = "/home/jitesh/3d/data/images_for_ndds_bg/collaged_images_random-size-v"
if not os.path.isdir(output):
    os.mkdir(output)
i = 0
# n=0

for n in tqdm(range(1000), desc="Generating images"):
    holder = []
    for x in range(randint(1, 7)):
        _holder = []
        for y in range(randint(2, 7)):
            _holder.append(images[i])
            if i + 1 < len(images):
                i += 1
            else:
                i = 0
                random.shuffle(images)
        holder.append(_holder)
    im_tile_resize = concat_tile_resize(holder)
    collage_image = im_tile_resize
    fin_width, fin_height = 1024, 1024

    is_width_bigger = False
    if collage_image.shape[1]/fin_width > collage_image.shape[0]/fin_height:
        is_width_bigger = True

    # print(is_width_bigger, collage_image.shape)
    if collage_image.shape[1] > fin_width and collage_image.shape[0] > fin_height:
        collage_image = collage_image[0:fin_height, 0:fin_width, :]
    elif is_width_bigger:
        collage_image = resize_img(
            src=collage_image, scale_percent=(fin_height + 1)/collage_image.shape[0]*100)
    else:
        collage_image = resize_img(
            src=collage_image, scale_percent=(fin_width + 1)/collage_image.shape[1]*100)
    collage_image = collage_image[0:fin_height, 0:fin_width, :]
    collage_image = resize_img(src=collage_image, size=(fin_width, fin_height))
    im_tile_resize = collage_image
    collaged_output = os.path.join(output, str(n+1)+".png")
    n += 1

    cv2.imwrite(collaged_output, collage_image)
# %%
