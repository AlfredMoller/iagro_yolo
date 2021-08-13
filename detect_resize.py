import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from PIL import Image

# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
output_path = './detections/resize/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')

# Initialize Flask application
app = Flask(__name__)

# API that returns JSON with classes found in images
@app.route('/detections', methods=['POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("images")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)

    num = 0

    # create list for final response
    response = []

    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time of detection: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    #remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404)


app.config['IMG_FOLDER'] = './detections/resize/'
app.config['IMG_RESIZED_RATIO'] = 500


# Function to img crop
def crop_img(img, img_name):
    w, h = img.size
    img_resized = None
    if w > h:
        left = (w - h) / 2
        right = left + h
        img_resized = img.crop((left, 0, right, h))
    elif h > w:
        top = (h - w) / 2
        bottom = top + w
        img_resized = img.crop((0, top, w, bottom))
    else:
        img_resized = img

    img_resized = resize_img(img_resized)
    img_resized.save(os.path.join(app.config['IMG_FOLDER'], img_name), 'JPEG', quality=90)

    to_return_path = os.path.join(app.config['IMG_FOLDER'], img_name)

    return to_return_path


# Function to img resized
def resize_img(img):
    size = img.size
    ratio = float(app.config['IMG_RESIZED_RATIO']) / max(size)
    new_img = tuple([int(x * ratio) for x in size])
    return img.resize(new_img, Image.ANTIALIAS)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename

    request_image = Image.open(image)
    print(image_name)
    # we made a resize and crop of the image to make a quick and proper detection
    output_img = Image.open(crop_img(request_image, image_name))
    image_np = load_image_into_numpy_array(output_img)

    image = Image.fromarray(image_np)
    lasted_filename = image_name.replace(".jpg", "") + '_output.jpg'
    image.save('./detections/resize/' + lasted_filename)

    # saving the resized image in a directory
    image.save(os.path.join(os.getcwd(), lasted_filename))
    img_raw = tf.image.decode_image(
        open(lasted_filename, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + lasted_filename, img)
    print('output saved to: {}'.format(output_path + lasted_filename))

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    result_name = lasted_filename


    # removing temporary image to save space for more images in the directory
    os.remove(output_path+image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)
if __name__ == '__main__':
    app.run(host="192.168.1.102",port=5000, debug=True)