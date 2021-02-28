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

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

import cv2

##################
##################

from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[2])
# print("ROOT_DIR: ", ROOT_DIR)

# Path to trained weights file
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')

parser.add_argument('--detect', required=False, default='rgbd+',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False,
                    default='/data/Akeaveny/Datasets/domain_adaptation/UMD/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')

parser.add_argument('--is_umd_real', required=False,
                    default=True,
                    type=bool,
                    metavar="/path/to/YCB/dataset/")

parser.add_argument('--save_inference_images', required=False,
                    default='/data/Akeaveny/Datasets/domain_adaptation/UMD/Real/test/pred_mask_rcnn/',
                    type=str,
                    metavar="/path/to/YCB/dataset/")

parser.add_argument('--num_frames', required=False, default=100,
                    type=int,
                    metavar='test or val')

parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/ or Logs and checkpoints directory (default=logs/)")

parser.add_argument('--show_plots', required=False,
                    default=True,
                    type=bool,
                    metavar='show plots from matplotlib')
parser.add_argument('--save_output', required=False,
                    default=False,
                    type=bool,
                    metavar='save terminal output to text file')

args = parser.parse_args()

############################################################
###########################################################

import dataset as UMD
save_to_folder = '/images/inspect_trained_model/'
MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
### config ###
MAX_GT_INSTANCES_ = 3
DETECTION_MAX_INSTANCES_ = 3
DETECTION_MIN_CONFIDENCE_ = 0.925
POST_NMS_ROIS_INFERENCE_ = 100
RPN_NMS_THRESHOLD_ = 0.8
DETECTION_NMS_THRESHOLD_ = 0.5
### crop ###
# IMAGE_RESIZE_MODE_ = "crop"
# IMAGE_MIN_DIM_ = 128
# IMAGE_MAX_DIM_ = 128
### sqaure ###
IMAGE_RESIZE_MODE_ = "square"
IMAGE_MIN_DIM_ = 640
IMAGE_MAX_DIM_ = 640

if not (os.path.exists(os.getcwd()+save_to_folder)):
    os.makedirs(os.getcwd()+save_to_folder)

###########################################################
###########################################################

if args.detect == 'rgb':
    from mrcnn import model as modellib, utils, visualize
if args.detect == 'rgbd':
    from mrcnn import modeldepth as modellib, utils, visualize
elif args.detect == 'rgbd+':
    from mrcnn import modeldepthv2 as modellib, utils, visualize
else:
    assert "*** No Model Selected ***"

###########################################################
###########################################################

def seq_get_masks(image, cur_detection, gt_mask, args):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']
    good_detect = False

    instance_masks = np.zeros((gt_mask.shape[0], gt_mask.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((gt_mask.shape[0], gt_mask.shape[1]), dtype=np.uint8)

    print("\tObject_ids", cur_class_ids)
    if cur_masks.shape[-1] > 0:

        for i in range(cur_masks.shape[-1]):
            good_detect = True

            cur_class_ids[i] = cur_class_ids[i]
            print("\tPred Affordance Label:", cur_class_ids[i])

            ### instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * cur_class_ids[i]
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    ########################
    #  add color to masks
    ########################
    instance_to_color = UMD.color_map()
    color_masks = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    return instance_masks, color_masks, good_detect

###########################################################
###########################################################

def crop_img(pil_img, crop_size, is_img=False):
    _dtype = np.array(pil_img).dtype
    pil_img = Image.fromarray(pil_img)
    crop_w, crop_h = crop_size
    img_width, img_height = pil_img.size
    left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
    top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
    pil_img = pil_img.crop((left, top, right, bottom))
    ###
    if is_img:
        img_channels = np.array(pil_img).shape[-1]
        img_channels = 3 if img_channels == 4 else img_channels
        resize_img = np.zeros((crop_w, crop_h, img_channels))
        resize_img[0:(bottom - top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
    else:
        resize_img = np.zeros((crop_w, crop_h))
        resize_img[0:(bottom - top), 0:(right - left)] = np.array(pil_img)
    ###
    return np.array(resize_img, dtype=_dtype)

###########################################################
###########################################################

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

    # select random test images
    np.random.seed(0)
    test_idx = np.random.choice(np.arange(0, len(dataset.image_ids), 1), size=int(args.num_frames), replace=False)
    # test_idx = np.random.choice(np.arange(0, len(dataset.image_ids), 1), size=int(len(dataset.image_ids)), replace=False)
    print("Chosen Files \n", len(test_idx))

    for num_image, idx in enumerate(test_idx):

        print("Running Inference on Image {} ..".format(num_image))

        ##############################
        #  Address for saving mask
        ##############################
        # for image_id in range(len(dataset.image_ids)):
        # print("\nimage_file:", dataset.image_reference(image_id))

        image_file1 = dataset.image_reference(idx)
        image_file2 = image_file1.split(args.dataset)[1]  # remove dataset path
        image_file3 = image_file2.split('.jpg')[0]
        idx = image_file3.split('rgb/')[1]

        ##############################
        ##############################
        if args.is_umd_real:
            rgb_addr = args.dataset     + 'Real/test/rgb/' + idx + '.jpg'
            depth_addr = args.dataset   + 'Real/test/depth/' + idx + '_depth.png'
            gt_mask_addr = args.dataset + 'Real/test/masks/' + idx + '_label.png'
        # print("gt_mask_addr: ", gt_mask_addr)

        if os.path.isfile(rgb_addr) == False:
            continue
        if os.path.isfile(depth_addr) == False:
            continue
        if os.path.isfile(gt_mask_addr) == False:
            continue

        gt_addr = args.save_inference_images + str(num_image) + '_gt.png'
        pred_addr = args.save_inference_images + str(num_image) + '_pred.png'
        colour_addr = args.save_inference_images + str(num_image) + '_colour.png'
        print("\tpred_mask_addr:", pred_addr)

        ##############################
        ### load
        ##############################

        rgb = np.array(skimage.io.imread(rgb_addr))
        depth = np.array(skimage.io.imread(depth_addr))
        gt_label = np.array(skimage.io.imread(gt_mask_addr))
        print("\tGT Affordance Label:", np.unique(gt_label))

        ##############################
        ### crop
        ##############################
        # print("GT SHAPE: ", gt_label.shape)
        #
        # resize = (int(640/3), int(480/3))
        # # rgb = cv2.resize(rgb, resize, interpolation=cv2.INTER_CUBIC)
        # # depth = cv2.resize(depth, resize, interpolation=cv2.INTER_NEAREST)
        # gt_label = cv2.resize(gt_label, resize, interpolation=cv2.INTER_NEAREST)
        #
        # crop_size = (128, 128)
        # # rgb = crop_img(rgb, crop_size=crop_size, is_img=True)
        # # depth = crop_img(depth,  crop_size=crop_size)
        # gt_label = crop_img(gt_label,  crop_size=crop_size)
        #
        # print("GT SHAPE: ", gt_label.shape)
        ######################
        # configure depth
        ######################

        depth[np.isnan(depth)] = 0
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        rgb, depth = rgb[..., :3], skimage.color.gray2rgb(depth)

        ##############################
        #  Detect
        ##############################

        if args.detect == 'rgb':
            ### load
            ### image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            ###     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            # run detect
            cur_detect = model.detect([rgb], verbose=0)[0]

        elif args.detect == 'rgbd' or args.detect == 'rgbd+':
            ### load
            ### image, depthimage, image_meta, gt_class_id, gt_bbox, gt_mask = \
            ###    modellib.load_images_gt(dataset, config, image_id, use_mini_mask=False)
            # run detect
            cur_detect = model.detectWdepth([rgb], [depth], verbose=0)[0]

        # get instance_masks
        instance_mask, color_mask, good_detect = seq_get_masks(rgb, cur_detect, gt_label, args)

        ####################

        cv2.imwrite(gt_addr, gt_label)
        cv2.imwrite(pred_addr, instance_mask)
        cv2.imwrite(colour_addr, color_mask)

        if args.show_plots:  # TODO: boolean string
            print("\tGT shape:", gt_label.shape)
            print("\tPred shape:", instance_mask.shape)

            masks = cur_detect['masks']
            # masks = gt_aff_mask
            class_ids = np.array(cur_detect['class_ids']) - 1
            ### print("class_ids:  ", class_ids)
            class_names = np.array(['grasp', 'cut', 'scoop', 'contain', 'pound', 'support', 'wrap-grasp'])
            visualize.display_instances(image=rgb, boxes=cur_detect['rois'], masks=masks,
                                        class_ids=class_ids, class_names=class_names,
                                        scores=cur_detect['scores'],
                                        title="Predictions",
                                        show_bbox=True, show_mask=True)
            # plt.tight_layout()
            # plt.show()

            ### plotting
            # cv2.imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            # cv2.imshow("depth", np.array(depth, dtype=np.uint8))
            # cv2.imshow("gt", gt_aff_mask * 25)
            # cv2.imshow("pred", instance_mask * 25)
            mask_file_path = os.getcwd() + save_to_folder + "pred.png"
            plt.savefig(mask_file_path, bbox_inches='tight')
            masked_image = cv2.imread(mask_file_path)
            cv2.imshow("masked_image", masked_image)
            cv2.waitKey(1)

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
