import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from matplotlib import pyplot as plt
from visualize import display_instances, display_top_masks
from utils import Dataset, extract_bboxes
from config import Config
from model import MaskRCNN
from model as modellib, utils
from PIL import Image, ImageDraw
ROOT_DIR = "/home/s/sqbqamar/Public/Plant/Fibre"


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

COCO_WEIGHTS_PATH= "/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Fibre/log/object20221110T1442/mask_rcnn_object_013.h5"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "log")


class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"
    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
   # NUM_CLASSES = 3  # Background + phone,laptop and mobile
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    
    LEARNING_RATE = 0.0001

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        super().__init__()
    
    
class InferenceConfig(Config):
    # Run detection on one image at a time
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    

config1 = InferenceConfig()


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            #print(mask_draw)
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids



def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CocoLikeDataset()
    dataset_train.load_data("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Fibre/Coco/train/_annotations.coco.json", "/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Fibre/Coco/train")
    dataset_train.prepare()
    #print(dataset_train)

    # Validation d"/home/s/sqbqamar/Public/Plant/Mask_RCNN/Datasets/train/demo.jsoataset
    dataset_val = CocoLikeDataset()
    dataset_val.load_data("/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Fibre/Coco/train/_annotations.coco.json", "/pfs/stor10/users/home/s/sqbqamar/Public/Plant/Fibre/Coco/train")
    dataset_val.prepare()
    
    
    #mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, calculate_map_at_every_X_epoch=2, verbose=1)
     
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    
    
   # augmentation = imgaug.augmenters.Sequential([imgaug.augmenters.Affine(rotate=(-45, 45))]))# custom_callbacks=[mean_average_precision_callback]) 
   #             augmentation = imgaug.augmenters.Sequential([ 
    #            imgaug.augmenters.Fliplr(1), 
     #           imgaug.augmenters.Flipud(1), 
      #          imgaug.augmenters.Affine(rotate=(-45, 45)), 
       #         imgaug.augmenters.Affine(rotate=(-90, 90)), 
        #        imgaug.augmenters.Affine(scale=(0.5, 1.5))]))
                


                
                
config = CustomConfig(num_classes=3)
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
#model_inference = modellib.MaskRCNN(mode="inference", config=config1, model_dir=DEFAULT_LOGS_DIR)
                                  

#custom_callbacks=[mean_average_precision_callback]

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)            
