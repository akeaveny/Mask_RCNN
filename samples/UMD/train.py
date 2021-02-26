"""
Mask R-CNN
Train on the Warehouse dataset and implement color splash effect.

Based on the work of Waleed Abdulla (Matterport)
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from imgaug import augmenters as iaa

############################################################
############################################################

from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[2])
# print("ROOT_DIR: ", ROOT_DIR)

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser( description='Train Mask R-CNN to detect Affordance.')

parser.add_argument('--train', required=False, default='rgb',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False,
                    default='/home/akeaveny/datasets/DomainAdaptation/UMD/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')

parser.add_argument('--display_keras', required=False, default=False,
                    type=str,
                    metavar='Display Keras Layers')

args = parser.parse_args()

############################################################
############################################################

import dataset as UMD

from mrcnn.config import Config
if args.train == 'rgb':
    from mrcnn import model as modellib, utils
elif args.train == 'rgbd':
    from mrcnn import modeldepth as modellib, utils
elif args.train == 'rgbd+':
    from mrcnn import modeldepthv2 as modellib, utils
else:
    print("*** No Model Selected ***")
    exit(1)

############################################################
#  train
############################################################

def train(model, args):

    """Train the model."""
    # Training dataset.
    dataset_train = UMD.UMDDataset()
    dataset_train.load_UMD(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UMD.UMDDataset()
    dataset_val.load_UMD(args.dataset, "val")
    dataset_val.prepare()

    if args.display_keras:
        model.keras_model.summary()
    config.display()

    ##################
    #  IMMGAUG
    ##################
    augmentation = iaa.Sometimes(0.833, iaa.Sequential([
        #########################
        # IMG & MASK
        #########################
        iaa.Fliplr(0.5),  # horizontal flips
        #########################
        # COLOR
        #########################
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.25)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True))  # apply augmenters in random order

    #############################
    #  Learning Rate Scheduler
    #############################
    # START = 0
    ### Training - Stage 1 HEADS
    ### HEADS
    print("\n************* trainining HEADS *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                # epochs=START + 20,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    ### Training - Stage 2a
    ### Finetune layers from ResNet stage 4 and up
    print("\n************* trainining ResNET 4+ *************")
    model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE/10,
              epochs=25,
              augmentation=augmentation,
              layers='4+')

    ### Training - Stage 3
    ### Fine tune all layers
    print("\n************* trainining ALL *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/100,
                epochs=30,
                augmentation=augmentation,
                layers='all')

############################################################
#  Training
############################################################

if __name__ == '__main__':
  
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = UMD.UMDConfig()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    train(model, args)
