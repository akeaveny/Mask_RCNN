import json
import glob
import cv2
import random

import os

import matplotlib.pyplot as plt

from PIL import Image # (pip install Pillow)

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing
stop = True
data = None

debug = True

###########################################################
# bbox
###########################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

###########################################################
# coco
###########################################################

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(class_id, sub_mask, mask, bbox):

    #################
    # mask contours
    #################

    h, w = sub_mask.size
    sub_mask = np.array(sub_mask.getdata(), dtype=np.uint8).reshape(w, h)

    sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_GRAY2BGR)
    sub_mask = cv2.cvtColor(sub_mask, cv2.COLOR_BGR2GRAY) * 255

    contours, hierarchy = cv2.findContours(sub_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #################
    # mask contours
    #################
    mask = np.array(mask, dtype=np.uint8)

    # (X coordinate value, Y coordinate value)
    cv2.rectangle(mask, (bbox[1], bbox[0]), (bbox[3], bbox[2]), 255, 2)

    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv2.drawContours(mask, contours, -1, 255, 3)

    #################
    #################

    region = {}
    region['region_attributes'] = {}
    region['shape_attributes'] = {}
    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["num_contours"] = len(contours)
    # region['shape_attributes']["all_points_x"] = np.array(x_list).tolist()
    # region['shape_attributes']["all_points_y"] = np.array(y_list).tolist()
    region['shape_attributes']["class_id"] = class_id

    for contour_idx, k in enumerate(contours):
        x_list = []
        y_list = []
        for i in k:
            for j in i:
                x_list.append(j[0])
                y_list.append(j[1])
        region['shape_attributes']["all_points_x" + str(contour_idx)] = np.array(x_list).tolist()
        region['shape_attributes']["all_points_y" + str(contour_idx)] = np.array(y_list).tolist()

    if VISUALIZE:
        # cv
        cv2.imshow("out", np.array(mask)*50)
        cv2.waitKey(0)
        # matplotlib
        # plt.imshow(label_img)
        # plt.plot(x_list, y_list, linewidth=1)
        # plt.show()
        # plt.ioff()

    return region

###########################################################
# Manual Config
###########################################################
np.random.seed(1)

dataset_name = 'UMD'

class_id = [0, 1, 2, 3, 4, 5, 6, 7]
print("Affordance IDs: \n{}\n".format(class_id))

######################
######################

json_path = '/home/akeaveny/git/Mask_RCNN/samples/UMD/json/Syn/'
json_name = 'coco_tools_'

data_path = '/home/akeaveny/datasets/DomainAdaptation/UMD/'
val_path = 'Syn/val/'
train_path = 'Syn/train/'
test_path = 'Syn/test/'

rgb_ext = '.png'
depth_ext = '_depth.png'
label_ext = '_label.png'

VISUALIZE = False

use_random_idx = True
num_val = 0
num_train = 4
num_test = 0

###########################################################
# JSON FILES
###########################################################

###################
# VAL
###################
if num_val == 0:
    print('******************** SKIPPING VAL ********************')
    pass
else:
    print('******************** VAL! ********************')
    folder_to_save = val_path
    rgb_path   = data_path + folder_to_save + 'rgb/' + '*' + rgb_ext
    depth_path = data_path + folder_to_save + 'depth/' + '*' + depth_ext
    label_path = data_path + folder_to_save + 'masks/' + '*' + label_ext

    print("labels: ", label_path)
    rgb_files = np.array(sorted(glob.glob(rgb_path)))
    depth_files = np.array(sorted(glob.glob(depth_path)))
    label_files = np.array(sorted(glob.glob(label_path)))
    assert (len(rgb_files) == len(depth_files) == len(label_files))
    print("Loaded label_files: ", len(label_files))

    if use_random_idx:
        val_idx = np.random.choice(np.arange(0, len(label_files), 1), size=int(num_val), replace=False)
        print("Chosen Files ", len(val_idx))
        rgb_files = rgb_files[val_idx]
        depth_files = depth_files[val_idx]
        label_files = label_files[val_idx]
    else:
        num_val = len(label_files)

    data = {}
    iteration = 0

    ###################
    ###################

    json_addr = json_path + json_name + 'val_' + np.str(len(label_files)) + '.json'
    print("json_addr: ", json_addr)
    for idx, label_file in enumerate(label_files):

        str_num = label_file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(label_ext)[0]

        print("rgb_file: ", rgb_files[idx].split(data_path)[1])
        print('Image: {}/{}'.format(iteration, len(label_files)))

        rgb_img   = np.array(cv2.imread(rgb_files[idx], -1))
        depth_img = np.array(cv2.imread(depth_files[idx], -1))
        label_img = np.array(cv2.imread(label_files[idx], -1))

        object_ids = np.unique(np.array(label_img))
        print("GT Affordances:", object_ids)

        # if VISUALIZE:
        #     cv2.imshow('rgb', rgb_img)
        #     cv2.imshow('obj_label', label_img * 40)
        #
        #     depth_img = depth_img / np.max(depth_img) * (2**8-1)
        #     depth_img = np.array(depth_img, dtype=np.uint8)
        #     cv2.imshow('depth', depth_img)
        #     cv2.imshow('heatmap', cv2.applyColorMap(depth_img, cv2.COLORMAP_JET))
        #     cv2.waitKey(1)

        ####################
        ### init
        ####################
        img_name = img_number + dataset_name
        data[img_name] = {}
        data[img_name]['fileref'] = ""
        data[img_name]['size'] = 640
        data[img_name]['filename'] = rgb_files[idx].split(data_path)[1]
        data[img_name]['depthfilename'] = depth_files[idx].split(data_path)[1]
        data[img_name]['base64_img_data'] = ""
        data[img_name]['file_attributes'] = {}

        ####################
        ### bbox
        ####################

        mask = np.array(label_img.copy(), dtype=np.uint8)

        # y1, x1, y2, x2
        bbox = extract_bboxes(mask)
        # (X coordinate value, Y coordinate value)
        # cv2.rectangle(mask, (bbox[1], bbox[0]), (bbox[3], bbox[2]), 255, 2)

        # data[img_name]['bbox'] = bbox.tolist()
        # data[img_name]['object_id'] = 1

        ###################
        # affmasks
        ###################
        data[img_name]['regions'] = {}
        regions = {}

        print("class ids: ", np.unique(label_img))
        label_img = Image.fromarray(label_img)

        ###
        sub_masks = create_sub_masks(label_img)
        for idx, sub_mask in sub_masks.items():
            if int(idx) > 0:
                object_id = int(idx)
                print("object_id: ", object_id)
                region = create_sub_mask_annotation(class_id=object_id,
                                                    sub_mask=sub_mask,
                                                    mask=label_img,
                                                    bbox=bbox)
                regions[np.str(object_id)] = region
        data[img_name]['regions'] = regions
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)

###################
# TRAIN
###################
if num_train == 0:
    print('******************** SKIPPING TRAIN ********************')
    pass
else:
    print('******************** TRAIN! ********************')
    folder_to_save = train_path
    rgb_path   = data_path + folder_to_save + 'rgb/' + '*' + rgb_ext
    depth_path = data_path + folder_to_save + 'depth/' + '*' + depth_ext
    label_path = data_path + folder_to_save + 'masks/' + '*' + label_ext

    print("labels: ", label_path)
    rgb_files = np.array(sorted(glob.glob(rgb_path)))
    depth_files = np.array(sorted(glob.glob(depth_path)))
    label_files = np.array(sorted(glob.glob(label_path)))
    assert (len(rgb_files) == len(depth_files) == len(label_files))
    print("Loaded label_files: ", len(label_files))

    if use_random_idx:
        train_idx = np.random.choice(np.arange(0, len(label_files), 1), size=int(num_train), replace=False)
        print("Chosen Files ", len(train_idx))
        rgb_files = rgb_files[train_idx]
        depth_files = depth_files[train_idx]
        label_files = label_files[train_idx]
    else:
        num_train = len(label_files)

    data = {}
    iteration = 0

    ###################
    ###################

    json_addr = json_path + json_name + 'train_' + np.str(len(label_files)) + '.json'
    print("json_addr: ", json_addr)
    for idx, label_file in enumerate(label_files):

        str_num = label_file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(label_ext)[0]

        print("rgb_file: ", rgb_files[idx].split(data_path)[1])
        print('Image: {}/{}'.format(iteration, len(label_files)))

        rgb_img   = np.array(cv2.imread(rgb_files[idx], -1))
        depth_img = np.array(cv2.imread(depth_files[idx], -1))
        label_img = np.array(cv2.imread(label_files[idx], -1))

        object_ids = np.unique(np.array(label_img))
        print("GT Affordances:", object_ids)

        # if VISUALIZE:
        #     cv2.imshow('rgb', rgb_img)
        #     cv2.imshow('obj_label', label_img * 40)
        #
        #     depth_img = depth_img / np.max(depth_img) * (2**8-1)
        #     depth_img = np.array(depth_img, dtype=np.uint8)
        #     cv2.imshow('depth', depth_img)
        #     cv2.imshow('heatmap', cv2.applyColorMap(depth_img, cv2.COLORMAP_JET))
        #     cv2.waitKey(1)

        ####################
        ### init
        ####################
        img_name = img_number + dataset_name
        data[img_name] = {}
        data[img_name]['fileref'] = ""
        data[img_name]['size'] = 640
        data[img_name]['filename'] = rgb_files[idx].split(data_path)[1]
        data[img_name]['depthfilename'] = depth_files[idx].split(data_path)[1]
        data[img_name]['base64_img_data'] = ""
        data[img_name]['file_attributes'] = {}

        ####################
        ### bbox
        ####################

        mask = np.array(label_img.copy(), dtype=np.uint8)

        # y1, x1, y2, x2
        bbox = extract_bboxes(mask)
        # (X coordinate value, Y coordinate value)
        # cv2.rectangle(mask, (bbox[1], bbox[0]), (bbox[3], bbox[2]), 255, 2)

        # data[img_name]['bbox'] = bbox.tolist()
        # data[img_name]['object_id'] = 1

        ###################
        # affmasks
        ###################
        data[img_name]['regions'] = {}
        regions = {}

        print("class ids: ", np.unique(label_img))
        label_img = Image.fromarray(label_img)

        ###
        sub_masks = create_sub_masks(label_img)
        for idx, sub_mask in sub_masks.items():
            if int(idx) > 0:
                object_id = int(idx)
                print("object_id: ", object_id)
                region = create_sub_mask_annotation(class_id=object_id,
                                                    sub_mask=sub_mask,
                                                    mask=label_img,
                                                    bbox=bbox)
                regions[np.str(object_id)] = region
        data[img_name]['regions'] = regions
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)