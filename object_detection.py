import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json

from io import StringIO
from PIL import Image
from watson_developer_cloud import VisualRecognitionV3

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Replace with your api key
visual_recognition = VisualRecognitionV3('2016-05-20', api_key='API_KEY')

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.6

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'w']

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

print('Downloading model... (This may take over 5 minutes)')
# Download model if not already downloaded
if not os.path.exists(PATH_TO_CKPT):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print('Extracting...')
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
else:
    print('Model already downloaded')

# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

# Path to test image, "test_image/image1.jpg"
TEST_IMAGE_PATH = 'test_image/image1.jpg'

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image = Image.open(TEST_IMAGE_PATH)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        # Create figure and axes
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(image_np)

        (height, width, x) = image_np.shape

        for i in range(0, int(min(num_detections, MAX_NUMBER_OF_BOXES))):
            score = np.squeeze(scores)[i]

            if score < MINIMUM_CONFIDENCE:
                break
            box = np.squeeze(boxes)[i]

            box_x = box[1] * width
            box_y = box[0] * height
            box_width = (box[3] - box[1]) * width
            box_height = (box[2] - box[0]) * height

            box_x2 = box[3] * width
            box_y2 = box[2] * height

            img2 = image.crop((box_x, box_y, box_x2, box_y2))

            path = 'cropped/image1'
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, 'img{}.jpg'.format(i))
            img2.save(full_path)

            # Classify images with watson visual recognition
            with open(full_path, 'rb') as images_file:
                results = visual_recognition.classify(images_file=images_file, threshold=0.7, classifier_ids=['default'])
                print(json.dumps(results, indent=2))
                label = results['images'][0]['classifiers'][0]['classes'][0]['class']
                ax.text(box_x + 5, box_y - 5, label, fontsize=10, color='white', bbox={'facecolor':COLORS[i % 8], 'edgecolor':'none'})

            # Create a Rectangle patch
            rect = patches.Rectangle((box_x, box_y), box_width, box_height, linewidth=2, edgecolor=COLORS[i % 8], facecolor='none')
            ax.add_patch(rect)

        plt.show()
