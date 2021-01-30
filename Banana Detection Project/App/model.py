import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
from skimage import io
import random
import colorsys
import cv2
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

print('Loading')

ROOT_DIR = './project'
UTILS = utils


class DetectorConfig(Config):    
    NAME = 'banana'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 fruit class
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 25


new_model_path = './model_512_8_101.h5'


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode

new_model = modellib.MaskRCNN(mode='inference', config=inference_config, model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert new_model_path != "", "Provide path to trained weights"
print("Loading weights from ", new_model_path)
new_model.load_weights(new_model_path, by_name=True)

# set color for class


def get_colors_for_class_ids(class_ids):
    class_ids = [x - 1 for x in class_ids]
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


def load_image(path):
    image = io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
      image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
      image = image[..., :3]
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            min_scale=inference_config.IMAGE_MIN_SCALE,
            max_dim=inference_config.IMAGE_MAX_DIM,
            mode=inference_config.IMAGE_RESIZE_MODE)
    return image


def display_image(image, figsize=(16, 16), ax=None):
    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Show area outside image boundaries:
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    ax.imshow(masked_image.astype(np.uint8))
    # plt.show()


def predict(picture):
    new_image = picture
    results = new_model.detect([new_image])
    r = results[0]
    visualize.display_instances(new_image, r['rois'], r['masks'], r['class_ids'], ['BG', 'banana'], r['scores'],
                                colors=get_colors_for_class_ids(r['class_ids']))
    return r


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def custom_display(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    # Number of instances
    N = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = colors or random_colors(N)
    height, width = image.shape[:2]
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    return image


def predict_live(picture):
    new_image = picture
    results = new_model.detect([new_image])
    r = results[0]
    result = custom_display(new_image, r['rois'], r['masks'], r['class_ids'], ['BG', 'banana'], r['scores'],
                            colors=get_colors_for_class_ids(r['class_ids']))
    return result


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        x = load_image('./IMG_0895.JPG')
    else:
        x = load_image(sys.argv[1])
    print(type(x), x.dtype, x.shape)
    print(predict(load_image('./IMG_0895.JPG')))
