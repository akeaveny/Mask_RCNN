"""
Mask R-CNN for Object_RPE
------------------------------------------------------------
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

import cv2

#########################
#  CRF
#########################
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
print("ROOT_DIR: ", ROOT_DIR)

# Path to trained weights file
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')

parser.add_argument('--detect', required=False, default='rgb',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/real/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='scissors_20k',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')

parser.add_argument('--weights', required=False,
                    default='/home/akeaveny/catkin_ws/src/object-rpe-ak/Mask_RCNN/logs/affordance20200928T2053/mask_rcnn_affordance_0012.h5',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/ or Logs and checkpoints directory (default=logs/)")

parser.add_argument('--post_process', required=False, default=True,
                    type=bool,
                    metavar='use CRFs')

parser.add_argument('--show_plots', required=False, default=True,
                    type=bool,
                    metavar='show plots from matplotlib')
parser.add_argument('--save_output', required=False, default=False,
                    type=bool,
                    metavar='save terminal output to text file')

args = parser.parse_args()


######################################
# ### crop ###
# # MAX_GT_INSTANCES_ = 3
# # DETECTION_MAX_INSTANCES_ = 30
# # DETECTION_MIN_CONFIDENCE_ = 0.5
# # RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
# # IMAGE_RESIZE_MODE_ = "crop"
# # IMAGE_MIN_DIM_ = 448
# # IMAGE_MAX_DIM_ = 448
# ### sqaure ###
# MAX_GT_INSTANCES_ = 3
# DETECTION_MAX_INSTANCES_ = 30
# DETECTION_MIN_CONFIDENCE_ = 0.5
# RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
# IMAGE_RESIZE_MODE_ = "square"
# IMAGE_MIN_DIM_ = 640
# IMAGE_MAX_DIM_ = 640
######################################

############################################################
#  REAL OR SYN
############################################################
# assert args.dataset_type == 'real' or args.dataset_type == 'syn' or args.dataset_type == 'syn1' or args.dataset_type == 'hammer' or args.dataset_type == 'hammer1'
if args.dataset_type == 'real':
    import dataset_real as UMD
    save_to_folder = '/images/test_images_real/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn':
    import dataset_syn as UMD
    save_to_folder = '/images/test_images_syn/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    ### crop ###
    MAX_GT_INSTANCES_ = 3
    DETECTION_MAX_INSTANCES_ = 30
    DETECTION_MIN_CONFIDENCE_ = 0.5
    CROP = True
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # MAX_GT_INSTANCES_ = 3
    # DETECTION_MAX_INSTANCES_ = 30
    # DETECTION_MIN_CONFIDENCE_ = 0.5
    # RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn1':
    import dataset_syn1 as UMD
    save_to_folder = '/images/test_images_syn1/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.9
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'hammer':
    import objects.dataset_syn_hammer as Affordance
    save_to_folder = '/images/objects/test_images_syn_hammer/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'hammer1':
    import objects.dataset_syn_hammer1 as Affordance
    save_to_folder = '/images/objects/test_images_syn_hammer1/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 896
    # IMAGE_MAX_DIM_ = 896
elif args.dataset_type == 'scissors':
    import objects.dataset_syn_scissors as Affordance
    save_to_folder = '/images/objects/test_images_syn_scissors/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'scissors_20k':
    import objects.dataset_syn_scissors_20k as Affordance
    save_to_folder = '/images/objects/test_images_syn_scissors_20k/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 10
    DETECTION_MAX_INSTANCES_ = 10
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640

if not (os.path.exists(os.getcwd()+save_to_folder)):
    os.makedirs(os.getcwd()+save_to_folder)

from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf

if args.detect == 'rgbd':
    from mrcnn import modeldepth as modellib, utils, visualize

###########################################################
# Test
###########################################################

def seq_get_masks(image, cur_detection, gt_mask, args):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    print("object_ids", cur_class_ids)
    if cur_masks.shape[-1] > 0:

        for i in range(cur_masks.shape[-1]):

            if args.dataset_type == 'real' or args.dataset_type == 'syn1':
                cur_class_ids[i] = cur_class_ids[i]
            elif args.dataset_type == 'syn' or args.dataset_type == 'hammer' or args.dataset_type == 'hammer1' or args.dataset_type == 'scissors':
                cur_class_ids[i] = map_affordance_label(cur_class_ids[i])
            print("Pred Affordance Label:", cur_class_ids[i])

            ### instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * cur_class_ids[i]
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    ########################
    #  add color to masks
    ########################
    instance_to_color = UMD.color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    return instance_masks, color_masks

def detect_and_get_masks(model, config, args):

    ########################
    #  Load test images
    ########################

    print("args.dataset_split", args.dataset_split)
    dataset = UMD.UMDDataset()
    dataset.load_UMD(args.dataset, args.dataset_split)
    dataset.prepare()

    config.display()

    print("Num of Test Images: {}".format(len(dataset.image_ids)))

    for image_id in range(len(dataset.image_ids)):

        print("\nimage_file:", dataset.image_reference(image_id))

        ##############################
        #  Address for saving mask
        ##############################

        image_file1 = dataset.image_reference(image_id)
        image_file2 = image_file1.split(args.dataset)[1]  # remove dataset path
        idx = image_file2.split('_rgb')[0] # remove _rgb label

        rgb_addr = args.dataset + idx + '_rgb.jpg'
        depth_addr = args.dataset + idx + '_depth.png'
        gt_mask_addr = args.dataset + idx + '_label.png'
        # gt_mask_addr = args.dataset + idx + '_gt_affordance.png'

        if os.path.isfile(rgb_addr) == False:
            continue;
        if os.path.isfile(depth_addr) == False:
            continue;
        if os.path.isfile(gt_mask_addr) == False:
            continue;

        mask_addr = args.dataset + idx + '_mask_og.png'
        color_mask_addr = args.dataset + idx + '_mask_color.png'
        cropped_mask_addr = args.dataset + idx + '_mask_cropped.png'
        print("mask_addr:", mask_addr)

        ##############################
        ### ground truth
        ##############################

        rgb = np.array(skimage.io.imread(rgb_addr))
        depth = np.array(skimage.io.imread(depth_addr))
        gt_label = np.array(skimage.io.imread(gt_mask_addr))
        ### print("GT RGB SHAPE: ", rgb.shape)
        print("GT Affordance Label:", np.unique(gt_label))

        ######################
        # configure depth
        ######################

        UMD_DEPTH_MAX = 3626

        depth[np.isnan(depth)] = 0
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0

        # convert to 8-bit image
        # depth = depth * (2 ** 16 -1) / np.max(depth)  ### 16 bit
        depth = depth * (2 ** 8 - 1) / UMD_DEPTH_MAX  ### 8 bit
        depth = np.array(depth, dtype=np.uint8)

        # print("depth min: ", np.min(np.array(depth)))
        # print("depth max: ", np.max(np.array(depth)))
        #
        # print("depth type: ", depth.dtype)
        # print("depth shape: ", depth.shape)

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        rgb, depth = rgb[..., :3], skimage.color.gray2rgb(depth)

        ##############################
        #  Resize
        ##############################
        ### rgb = cv2.resize(rgb, dsize=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM), interpolation=cv2.INTER_CUBIC)
        ### depth = cv2.resize(depth, dsize=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM), interpolation=cv2.INTER_CUBIC)
        ### gt_label = cv2.resize(gt_label, dsize=(1280, 1280), interpolation=cv2.INTER_CUBIC)

        ##############################
        #  CROP
        ##############################

        if CROP == True:
            # Pick a random crop
            h, w = rgb.shape[:2]

            x = (w - config.IMAGE_MIN_DIM) // 2
            y = (h - config.IMAGE_MIN_DIM) // 2

            rgb = rgb[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            depth = depth[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            gt_label = gt_label[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
        # print("gt_label: ", gt_label.shape)

        ##############################
        #  Detect
        ##############################

        if args.detect == 'rgb':
            # run detect
            cur_detect = model.detect([rgb], verbose=0)[0]

        elif args.detect == 'rgbd':
            # run detect
            cur_detect = model.detectWdepth([rgb], [depth], verbose=0)[0]

        # get instance_masks
        instance_mask, color_mask = seq_get_masks(rgb, cur_detect, gt_label, args)

        ####################

        cv2.imwrite(mask_addr, instance_mask)
        cv2.imwrite(color_mask_addr, color_mask)
        cv2.imwrite(cropped_mask_addr, gt_label)

        ####################
        if args.post_process:
            img = rgb
            # labels, n_labels = instance_mask, config.NUM_CLASSES
            labels, n_labels = gt_label, config.NUM_CLASSES

            # Example using the DenseCRF2D code
            d = dcrf.DenseCRF(img.shape[0] * img.shape[1], n_labels)

            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=0)
            d.setUnaryEnergy(U)

            # This potential penalizes small pieces of segmentation that are
            # spatially isolated -- enforces more spatially consistent segmentations
            feats = create_pairwise_gaussian(sdims=(30, 30), shape=img.shape[:2])
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features --
            # because the segmentation that we get from CNN are too coarse
            # and we can use local color features to refine them
            feats = create_pairwise_bilateral(sdims=(3, 3), schan=(3, 3, 3),
                                              img=img, chdim=2)
            d.addPairwiseEnergy(feats, compat=10,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            Q = d.inference(5)
            crf = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
            crf = np.array(crf, dtype=np.uint8)

            cv2.imwrite(mask_addr, crf)

            if args.show_plots:
                print("GT shape:", gt_label.shape)
                print("Pred shape:", instance_mask.shape)
                print("resize_pred shape:", instance_mask.shape)

                cv2.imshow("rgb", img)
                cv2.imshow("gt", gt_label * 100)
                cv2.imshow("pred", labels * 100)
                cv2.imshow("crf", crf * 100)
                cv2.waitKey(0)

        ####################
        else:
            if args.show_plots:
                print("GT shape:", gt_label.shape)
                print("Pred shape:", instance_mask.shape)
                print("resize_pred shape:", instance_mask.shape)

                cv2.imshow("gt", gt_label * 25)
                cv2.imshow("resize pred", instance_mask * 25)
                cv2.waitKey(0)

###########################################################
# FOR SYTHENTIC IMAGES
# LOOKUP FROM OBJECT ID TO AFFORDANCE LABEL
###########################################################
def map_affordance_label(current_id):

    # 1
    grasp = [
        20,  # "hammer-grasp"
        22,  # "hammer-grasp"
        24,  # "hammer-grasp"
        26,  # "hammer-grasp"
        #
        28,  # "knife-grasp"
        30,  # "knife-grasp"
        32,  # "knife-grasp"
        34,  # "knife-grasp"
        36,  # "knife-grasp"
        38,  # "knife-grasp"
        40,  # "knife-grasp"
        42,  # "knife-grasp"
        44,  # "knife-grasp"
        46,  # "knife-grasp"
        48,  # "knife-grasp"
        50,  # "knife-grasp"
        #
        52,  # "ladle-grasp"
        54,  # "ladle-grasp"
        56,  # "ladle-grasp"
        #
        58,  # "mallet-grasp"
        60,  # "mallet-grasp"
        62,  # "mallet-grasp"
        64,  # "mallet-grasp"
        #
        66,  # "mug-grasp"
        69,  # "mug-grasp"
        72,  # "mug-grasp"
        75,  # "mug-grasp"
        78,  # "mug-grasp"
        81,  # "mug-grasp"
        84,  # "mug-grasp"
        87,  # "mug-grasp"
        90,  # "mug-grasp"
        93,  # "mug-grasp"
        96,  # "mug-grasp"
        99,  # "mug-grasp"
        102,  # "mug-grasp"
        105,  # "mug-grasp"
        108,  # "mug-grasp"
        111,  # "mug-grasp"
        114,  # "mug-grasp"
        117,  # "mug-grasp"
        120,  # "mug-grasp"
        123,  # "mug-grasp"
        #
        130,  # "saw-grasp"
        132,  # "saw-grasp"
        134,  # "saw-grasp"
        #
        136,  # "scissors-grasp"
        138,  # "scissors-grasp"
        140,  # "scissors-grasp"
        142,  # "scissors-grasp"
        144,  # "scissors-grasp"
        146,  # "scissors-grasp"
        148,  # "scissors-grasp"
        150,  # "scissors-grasp"
        #
        152,  # "scoop-grasp"
        154,  # "scoop-grasp"
        #
        156,  # "shears-grasp"
        158,  # "shears-grasp"
        #
        160,  # "shovel-grasp"
        162,  # "shovel-grasp"
        #
        164,  # "spoon-grasp"
        166,  # "spoon-grasp"
        168,  # "spoon-grasp"
        170,  # "spoon-grasp"
        172,  # "spoon-grasp"
        174,  # "spoon-grasp"
        176,  # "spoon-grasp"
        178,  # "spoon-grasp"
        180,  # "spoon-grasp"
        182,  # "spoon-grasp"
        #
        184,  # "tenderizer-grasp"
        #
        186,  # "trowel-grasp"
        188,  # "trowel-grasp"
        190,  # "trowel-grasp"
        #
        192,  # "turner-grasp"
        194,  # "turner-grasp"
        196,  # "turner-grasp"
        198,  # "turner-grasp"
        200,  # "turner-grasp"
        202,  # "turner-grasp"
        204,  # "turner-grasp"
    ]

    # 2
    cut = [
        28 + 1,  # "knife-cut"
        30 + 1,  # "knife-cut"
        32 + 1,  # "knife-cut"
        34 + 1,  # "knife-cut"
        36 + 1,  # "knife-cut"
        38 + 1,  # "knife-cut"
        40 + 1,  # "knife-cut"
        42 + 1,  # "knife-cut"
        44 + 1,  # "knife-cut"
        46 + 1,  # "knife-cut"
        48 + 1,  # "knife-cut"
        50 + 1,  # "knife-cut"
        #
        130 + 1,  # "saw-cut"
        132 + 1,  # "saw-cut"
        134 + 1,  # "saw-cut"
        #
        136 + 1,  # "scissors-cut"
        138 + 1,  # "scissors-cut"
        140 + 1,  # "scissors-cut"
        142 + 1,  # "scissors-cut"
        144 + 1,  # "scissors-cut"
        146 + 1,  # "scissors-cut"
        148 + 1,  # "scissors-cut"
        150 + 1,  # "scissors-cut"
        #
        156 + 1, # "shears-cut"
        158 + 1,
    ]

    # 3
    scoop = [
        152 + 1,  # "scoop-scoop"
        154 + 1,  # "scoop-scoop"
        #
        160 + 1,  # "shovel-scoop"
        162 + 1,  # "shovel-scoop"
        #
        164 + 1,  # "spoon-scoop"
        166 + 1,  # "spoon-scoop"
        168 + 1,  # "spoon-scoop"
        170 + 1,  # "spoon-scoop"
        172 + 1,  # "spoon-scoop"
        174 + 1,  # "spoon-scoop"
        176 + 1,  # "spoon-scoop"
        178 + 1,  # "spoon-scoop"
        180 + 1,  # "spoon-scoop"
        182 + 1,  # "spoon-scoop"
        #
        186 + 1,  # "trowel-scoop"
        188 + 1,  # "trowel-scoop"
        190 + 1,  # "trowel-scoop"
    ]

    # 4
    contain = [
        1,  # "bowl-contain"
        2,  # "bowl-contain"
        3,  # "bowl-contain"
        4,  # "bowl-contain"
        5,  # "bowl-contain"
        6,  # "bowl-contain"
        7,  # "bowl-contain"
        #
        8,  # "cup-contain"
        10,  # "cup-contain"
        12,  # "cup-contain"
        14,  # "cup-contain"
        16,  # "cup-contain"
        18,  # "cup-contain"
        #
        52 + 1,  # "ladle-contain"
        54 + 1,  # "ladle-contain"
        56 + 1,  # "ladle-contain"
        66 + 1,  # "mug-contain"
        69 + 1,  # "mug-contain"
        72 + 1,  # "mug-contain"
        75 + 1,  # "mug-contain"
        78 + 1,  # "mug-contain"
        81 + 1,  # "mug-contain"
        84 + 1,  # "mug-contain"
        87 + 1,  # "mug-contain"
        90 + 1,  # "mug-contain"
        93 + 1,  # "mug-contain"
        96 + 1,  # "mug-contain"
        99 + 1,  # "mug-contain"
        #
        102 + 1,  # "mug-contain"
        105 + 1,  # "mug-contain"
        108 + 1,  # "mug-contain"
        111 + 1,  # "mug-contain"
        114 + 1,  # "mug-contain"
        117 + 1,  # "mug-contain"
        120 + 1,  # "mug-contain"
        123 + 1,  # "mug-contain"
        #
        126,  # "pot-contain"
        128,  # "pot-contain"
    ]

    # 5
    pound = [
        20 + 1, #"hammer-pound"
        22 + 1, #"hammer-pound"
        24 + 1, #"hammer-pound"
        26 + 1, #"hammer-pound"
        #
        58 + 1, #'mallet-pound'
        60 + 1, #'mallet-pound'
        62 + 1, #'mallet-pound'
        64 + 1, #'mallet-pound'
        #
        184 + 1, #'tenderizer-pound'
    ]

    # 6
    support = [
        192 + 1,  # "turner-support"
        194 + 1,  # "turner-support"
        196 + 1,  # "turner-support"
        198 + 1,  # "turner-support"
        200 + 1,  # "turner-support"
        202 + 1,  # "turner-support"
        204 + 1,  # "turner-support"
    ]

    # 7
    wrap_grasp = [
        8 + 1, # "cup-wrap_grasp"
        10 + 1, # "cup-wrap_grasp"
        12 + 1, # "cup-wrap_grasp"
        14 + 1, # "cup-wrap_grasp"
        16 + 1, # "cup-wrap_grasp"
        18 + 1, # "cup-wrap_grasp"
        #
        66 + 2, # "mug-wrap_grasp"
        69 + 2, # "mug-wrap_grasp"
        72 + 2, # "mug-wrap_grasp"
        75 + 2, # "mug-wrap_grasp"
        78 + 2, # "mug-wrap_grasp"
        81 + 2, # "mug-wrap_grasp"
        84 + 2, # "mug-wrap_grasp"
        87 + 2, # "mug-wrap_grasp"
        90 + 2, # "mug-wrap_grasp"
        93 + 2, # "mug-wrap_grasp"
        96 + 2, # "mug-wrap_grasp"
        99 + 2, # "mug-wrap_grasp"
        102 + 2, # "mug-wrap_grasp"
        105 + 2, # "mug-wrap_grasp"
        108 + 2, # "mug-wrap_grasp"
        111 + 2, # "mug-wrap_grasp"
        114 + 2, # "mug-wrap_grasp"
        117 + 2, # "mug-wrap_grasp"
        120 + 2, # "mug-wrap_grasp"
        123 + 2, # "mug-wrap_grasp"
        #
        126 + 1, # "pot-wrap_grasp"
        128 + 1, # "pot-wrap_grasp"
    ]

    if current_id in grasp:
        return 1
    elif current_id in cut:
        return 2
    elif current_id in scoop:
        return 3
    elif current_id in contain:
        return 4
    elif current_id in pound:
        return 5
    elif current_id in support:
        return 6
    elif current_id in wrap_grasp:
        return 7
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        print(current_id)
        exit(1)

###########################################################
# 
###########################################################
if __name__ == '__main__':


    class InferenceConfig(UMD.UMDConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        MEAN_PIXEL = MEAN_PIXEL_
        USE_MINI_MASK = False
        RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES_
        IMAGE_RESIZE_MODE = IMAGE_RESIZE_MODE_
        IMAGE_MIN_DIM = IMAGE_MIN_DIM_
        IMAGE_MAX_DIM = IMAGE_MAX_DIM_
        MAX_GT_INSTANCES = MAX_GT_INSTANCES_
        DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES_
        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE_
        POST_NMS_ROIS_INFERENCE = POST_NMS_ROIS_INFERENCE_
        RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD_
        DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD_
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    detect_and_get_masks(model, config, args)
