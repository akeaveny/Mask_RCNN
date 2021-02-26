import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log

# ###########################################################
# # Dataset
# ###########################################################

class UMDConfig(Config):
    """Configuration for training on the toy  dataset.
    # Derives from the base Config class and overrides some values.
    # """
    # Give the configuration a recognizable name
    NAME = "UMD"

    NUM_CLASSES = 1 + 7 # Number of classes (including background)

    ##################################
    ###  GPU
    ##################################

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    bs = GPU_COUNT * IMAGES_PER_GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ##################################
    ###  Backbone
    ##################################

    ### BACKBONE = "resnet50"

    ##################################
    ###
    ##################################

    LEARNING_RATE = 1e-03
    WEIGHT_DECAY = 1e-04

    ##################################
    ###  NUM OF IMAGES
    ##################################

    # Number of training steps per epoch
    STEPS_PER_EPOCH = (5000) // bs
    VALIDATION_STEPS = (1250) // bs

    ##################################
    ###  FROM DATASET STATS
    ##################################
    ''' --- run datasetstats for all params below --- '''

    MEAN_PIXEL = np.array([91.15, 88.89, 98.80])  ### REAL

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024
    RPN_ANCHOR_SCALES = (8, 16, 24, 32, 64)  ### 1024

    # IMAGE_RESIZE_MODE = "square"
    # IMAGE_MIN_DIM = 640
    # IMAGE_MAX_DIM = 640
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    # MASK_SHAPE = [56, 56]  # TODO: AFFORANCENET TRIED 14, 28, 56, 112, 224

    MAX_GT_INSTANCES = 20  # really only have 1 obj/image or max 3 labels/object
    DETECTION_MAX_INSTANCES = 20

    DETECTION_MIN_CONFIDENCE = 0.9

    TRAIN_ROIS_PER_IMAGE = 100  # TODO: DS bowl 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

# ###########################################################
# # Dataset
# ###########################################################

class UMDDataset(utils.Dataset):

    def load_UMD(self, dataset_dir, subset):
        """Load a subset of the UMD dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        #   1 - 'grasp'
        #   2 - 'cut'
        #   3 - 'scoop'
        #   4 - 'contain'
        #   5 - 'pound'
        #   6 - 'support'
        #   7 - 'wrap-grasp'
        self.add_class("UMD", 1, "grasp")
        self.add_class("UMD", 2, "cut")
        self.add_class("UMD", 3, "scoop")
        self.add_class("UMD", 4, "contain")
        self.add_class("UMD", 5, "pound")
        self.add_class("UMD", 6, "support")
        self.add_class("UMD", 7, "wrap-grasp")


        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        if subset == 'train':
             annotations = {}
             print("------------------LOADING TRAIN!------------------")
             annotations.update(json.load(
               open('/home/akeaveny/git/Mask_RCNN/samples/UMD/json/Real/coco_tools_train_5000.json')))

        elif subset == 'val':
            annotations = {}
            print("------------------LOADING VAL!--------------------")
            annotations.update(json.load(
                open('/home/akeaveny/git/Mask_RCNN/samples/UMD/json/Real/coco_tools_val_1250.json')))

        elif subset == 'test':
            annotations = {}
            print("------------------LOADING Test!--------------------")
            annotations.update(json.load(
                open('/home/akeaveny/git/Mask_RCNN/samples/UMD/json/Real/coco_tools_val_4.json')))

        annotations = list(annotations.values())
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            print(image_path)  # TODO: print all files
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "UMD",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a UMD dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "UMD":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_IDs = np.zeros([len(info["polygons"])], dtype=np.int32)

        #################
        # tools
        #################

        # for i, p in enumerate(info["polygons"]):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1
        #     class_IDs[i] = p['class_id']

        #################
        # clutter
        #################

        for i, p in enumerate(info["polygons"]):
            for countour_idx, _ in enumerate(range(p["num_contours"])):
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y' + str(countour_idx)], p['all_points_x' + str(countour_idx)])
                mask[rr, cc, i] = 1
                class_IDs[i] = p['class_id']

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_IDs

    def load_image_rgb_depth(self, image_id):

        file_path = np.str(image_id).split("rgb.jpg")[0]

        rgb = skimage.io.imread(file_path + "rgb.jpg")
        depth = skimage.io.imread(file_path + "depth.png")

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        return rgb[..., :3], skimage.color.gray2rgb(depth)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "UMD":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
