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
from flask_bcrypt import Bcrypt
import bcrypt
import psycopg2 as psy
from psycopg2 import Error
import json

# Define flask app
app = Flask(__name__, static_url_path='/static')
app.config['IMG_FOLDER'] = 'static/output/'
app.config['IMG_RESIZED_RATIO'] = 500
bcrypt = Bcrypt(app)

#Provisory Params for Psql connection
host_con = 'ec2-52-0-114-209.compute-1.amazonaws.com'
port_con = '5432'
db_con = 'd9iu7pl5ikjcgh'
user_con = 'rkjkiitazamftd'
pass_con = 'ba53f6b72bc5f4dd0e2da437f05685416a7512cec11894065885561412448edb'

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
        print('time: {}'.format(t2 - t1))

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



#---------------------------------------------------Login Request---------------------------------------------------

@app.route('/login', methods=['GET'])
def login():
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'GET':

        username = request.form.get('username')
        password = request.form.get('password')
        print("usuario",username)
        print("pass",password)
        #username = request.form['username']
        #password = request.form['password']

        try:
            connection = psy.connect(host= host_con, port= port_con ,database=db_con, user=user_con , password=pass_con)

            cursor = connection.cursor()
            cursor.execute('SELECT password,name,lastname,id_user FROM accounts WHERE username = %s', (username,))
            row = cursor.fetchone()

            if row:
                msgkey = "Usuario Logeado"
                print(row[0])
                nombre = row[1]
                apellido = row[2]
                id_usu = row[3]
                nombcompleto = nombre + " " + apellido

                #   if bcrypt.check_password_hash(row[0],password) ==True:
                if bcrypt.check_password_hash(row[0], password):

                    print(msgkey)
                    msgok = "Usuario Logeado!"
                    return jsonify(message=msgok, nomb_usu=nombcompleto, id_usu=id_usu)

                elif not bcrypt.check_password_hash(row[0], password):

                    msgfail = "Usuario con Contraseña Incorrecta!"
                    print(msgfail)
                    return jsonify(message=msgfail, nomb_usu="", id_usu="")

            else:

                msgfail = "Usuario no se Encuentra Registrado!"
                print(msgfail)
                return jsonify(message=msgfail, nomb_usu="", id_usu="")

        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()

def lastrecord() -> int:
    sql = "SELECT MAX(id_user) from accounts"
    try:
        connection = psy.connect(host= host_con, port= port_con ,database=db_con, user=user_con , password=pass_con)
        cursor = connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        connection.commit()
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()
    return int(row[0])

#---------------------------------------------------User Record Request---------------------------------------------------

@app.route('/user_register', methods=['POST'])
def register():
    if request.method == 'POST':

        username = request.form['corr']
        password = request.form['clav']
        name = request.form['nomb']
        lastname = request.form['ape']
        telephone = request.form['tel']
        identif = request.form['ced']

        encryptpass = bcrypt.generate_password_hash(password).decode('utf-8')
        last_record = lastrecord()
        print("ulltimo registro guardado",last_record)

        try:
            connection = psy.connect(host= host_con, port= port_con ,database=db_con, user=user_con , password=pass_con)
            cursor = connection.cursor()

            cursor.callproc('check_usrexist',[username])
            account = cursor.fetchone()
            if account:
                msgexist = "Ya existe este Usuario!"
                print(msgexist)
                return jsonify(message=msgexist)

            else:

                cursor.callproc('insert_newusr', [username, encryptpass, name, lastname, telephone, identif])
                connection.commit()
                msgok = "Usuario registrado con Éxito!"
                print(msgok)
                return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Guardar!"
            return jsonify(message=msgfail)

        finally:
            connection.close()


#---------------------------------------------------Update User Record Request---------------------------------------------------

@app.route('/updt_user', methods=['POST'])
def updt_user_info():
    if request.method == 'POST':

        usr_id = request.form['iduser']
        usr_mail = request.form['username']
        usr_pass = request.form['password']
        usr_name = request.form['name']
        usr_last = request.form['lastname']
        usr_tel= request.form['telephone']
        usr_ident = request.form['identification']

        encryptpass = bcrypt.generate_password_hash(usr_pass).decode('utf-8')

        try:
            connection = psy.connect(host= host_con, port= port_con ,database=db_con, user=user_con , password=pass_con)
            cursor = connection.cursor()
            cursor.execute(""" UPDATE accounts SET username= %s, password= %s, name= %s, lastname =%s, telephone= %s, identification =%s  where accounts.id_user =%s  """, (usr_mail, encryptpass, usr_name, usr_last, usr_tel, usr_ident, usr_id))
            connection.commit()
            msgok = "Perfil de Usuario Actualizado!!!"
            print(msgok)
            return jsonify(message=msgok)

        except Error as error:
            print(error)
            msgfail = "Error al Actualizar Perfil de Usuario Actualizado!"
            return jsonify(message=msgfail)

        finally:
            connection.close()

#---------------------------------------------------List User Request---------------------------------------------------


@app.route('/list_user', methods=['GET'])
def list_user_info():
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'GET':

        #userid = request.form['id_user']
        userid = request.form.get('id_user')

        try:
            connection = psy.connect(host= host_con, port= port_con ,database= db_con, user= user_con, password= pass_con)
            cursor = connection.cursor()

            # -> Replaced cursor Query for store procedure to improve security at moment fo bringing datas
            cursor.callproc('get_usrdata', [userid])
            row = cursor.fetchone()

            user_mail = row[0]
            user_name = row[1]
            user_last = row[2]
            user_tel =  row[3]
            user_ident = row[4]

            return jsonify(mail=user_mail, name=user_name, lastn=user_last, telephone= user_tel, ident= user_ident)


        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()


def canalizar(request_image,image_name):

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
    lista_res = list()

    print('detections:')
    for i in range(nums[0]):
        res = dict(name=str(class_names[int(classes[0][i])]), scores= float( np.array(scores[0][i])  ) )
        lista_res.append(res)
    print(lista_res)
    #test = dict((k, v) for k, v in res.items() if v >= 5)
    #print(test)
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + lasted_filename, img)
    print('output saved to: {}'.format(output_path + lasted_filename))

    # prepare image for response
    _, img_encoded = cv2.imencode('.png', img)
    response = img_encoded.tostring()
    result_name = lasted_filename

    # removing temporary image to save space for more images in the directory
    os.remove(output_path + image_name)
    return response



# API that returns image with detections on it
@app.route('/image', methods= ['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    if image_name.endswith('.png'):
        image.save(output_path + image_name)

        im = Image.open(output_path+ image_name)
        rgb_im = im.convert('RGB')
        last = image_name[:4]
        nuevo = last + ".jpg"
        rgb_im.save(output_path + nuevo)

        os.remove(output_path+image_name)

        requ = Image.open(output_path + nuevo)
        lt = nuevo
        df= canalizar(requ,lt)

    else:
        request_image = Image.open(image)
        df= canalizar(request_image,image_name)

    try:
        return Response(response=df, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# Run server
if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5000, debug=True)