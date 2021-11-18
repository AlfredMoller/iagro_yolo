import time
from builtins import list
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
from PIL import Image
from flask_bcrypt import Bcrypt
import bcrypt
import time
import psycopg2 as psy
from psycopg2 import Error
from itertools import groupby
from dotenv import load_dotenv
from dbx_droptest import *
from det_authenticator import *
from flask_limiter import Limiter
import uuid


# Define flask app
app = Flask(__name__, static_url_path='/static')
app.config['IMG_FOLDER'] = 'static/output/'
app.config['IMG_RESIZED_RATIO'] = 500
app.config['SECRET_KEY'] = os.getenv("app_key")
bcrypt = Bcrypt(app)
#limiter = Limiter(app)
load_dotenv()

#Params for Psql connection
host_con = os.getenv("host_con")
port_con = os.getenv("port_con")
db_con = os.getenv("db_con")
user_con = os.getenv("user_con")
pass_con = os.getenv("pass_con")

#Params for DropBox image storage testing
dbox_key = os.getenv("dbox_key2")

# customize your API through the following parameters
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
tiny = False  # set to True if using a Yolov3 Tiny model
size = 416  # size images are resized to for model
output_path = './detections/resize/'  # path to output folder where images with detections are saved
num_classes = 80  # number of classes in model

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


# ---------------------------------------------------Pest Id stored Function---------------------------------------------------

def get_pest_id(name) -> int:
    row = None
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('get_pestId', [name])
        row = cursor.fetchone()
        # print("ID de Plaga:", row[0])
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()
    return int(row[0])


# ---------------------------------------------------Save History Header Function---------------------------------------------------

def save_history(list_partial):
    # We have to check the size of the sent element into this function -> then store in the DB
    try:
        latitude = "-26.5326457"
        longitude = "-57.0399334"
        description = "Infectado"
        city = "San Miguel"
        now = datetime.now()
        color = "#f3435"
        idus = 1
        stat = "Infectado"

        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()

        for lp in list_partial:
            print("probamos de nuevo:", lp.get('name'))
            cursor.callproc('save_mul_values',
                            [latitude, longitude, description, city, now.strftime('%y-%m-%d'), color, idus, stat])
            connection.commit()
            if len(list_partial) > 0:
                idhistory = lastrecord()
                print("id rescatado:", idhistory)
                name_onebyone = lp.get('name')
                save_history_detail(idhistory, name_onebyone)
            else:
                print("Error al tratar de insertar los datos en la BD")
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()


# ---------------------------------------------------Save History Detail Function---------------------------------------------------

def save_history_detail(idhistory, det_final):
    """-------------------Modified Version---------------------"""
    idpest = get_pest_id(det_final)
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('save_mul_dt', [idhistory, idpest])
        connection.commit()
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()


# ---------------------------------------------------Login Request---------------------------------------------------

@app.route('/login', methods=['POST'])
def login():
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST':

        usr_name = request.form['username']
        usr_pass = request.form['password']
        usr_ip = request.form['ipadress']

        try:
            connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
            cursor = connection.cursor()
            cursor.callproc('get_logus', [usr_name])
            row = cursor.fetchone()
            #chkd_ipnde= chk_ipornode_one(usr_ip)
           # if chkd_ipnde.get("status") == True:
            if chk_user_penalty(usr_ip, connection) != True:
                if row:
                    # with counter.get_lock():
                    nombcompleto = " ".join([row[1],row[2]])
                    uui_usr = row[3]
                    id_usu = row[4]

                    if bcrypt.check_password_hash(row[0], usr_pass):
                        auth_token = auth_builder(uui_usr)
                        #auth_token = auth_builder(bcrypt.generate_password_hash(uui_usr).decode('utf-8'))
                        status = True
                        count_s = count_status(status)
                        p1 = count_fail(count_s)
                        print("Id publico:", uui_usr)
                        print("nuevo uuid:",bcrypt.generate_password_hash(uui_usr).decode('utf-8'))
                        print(p1.get_pnlt_time())
                        #if current logged device is not the same as store previously -> Send message with bot
                        return jsonify(status=200, msg="Usuario Logeado!", nomb_usu=nombcompleto, id_usu=id_usu,
                                       token=auth_token)

                    elif not bcrypt.check_password_hash(row[0], usr_pass):
                        status = False
                        count_f = count_status(status)
                        p1 = count_fail(count_f)
                        print(p1.get_pnlt_time())
                        if p1.get_pnlt_time() == 3:
                            date_penalty = datetime.now()
                            sv_user_penalty(usr_ip, date_penalty, connection)
                            return jsonify(status=403, msg="Procederemos a darle una penalizacion de 3 minutos")
                        else:
                            return jsonify(status=401, msg="Usuario o clave incorrecta...Intentelo de nuevo!")
                else:
                    status = False
                    count_f = count_status(status)
                    p1 = count_fail(count_f)
                    print(p1.get_pnlt_time())
                    if p1.get_pnlt_time() == 3:
                        date_penalty = datetime.now()
                        sv_user_penalty(usr_ip, date_penalty, connection)
                        return jsonify(status= 403, msg="Procederemos a darle una penalizacion de 3 minutos!  ‾＼_(ツ)_／‾")
                    else:
                        return jsonify(status=404, msg="Usuario no se encuentra Registrado!")
            else:
                return jsonify(status=403, msg="Usted se encuentra con una penalizacion temporal!")

        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()


# ---------------------------------------------------Get Last Record Function---------------------------------------------------

def lastrecord() -> int:
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('get_lastId')
        row = cursor.fetchone()
        connection.commit()
    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()
    return int(row[0])


# ---------------------------------------------------User Record Request---------------------------------------------------

@app.route('/user_register', methods=['POST'])
def register():
    if request.method == 'POST':

        username = request.form['corr']
        password = request.form['clav']
        name = request.form['nomb']
        lastname = request.form['ape']
        telephone = request.form['tel']
        identif = request.form['ced']
        user_uuid = str(uuid.uuid4())

        encryptpass = bcrypt.generate_password_hash(password).decode('utf-8')

        try:
            connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
            cursor = connection.cursor()
            cursor.callproc('check_usrexist', [username])
            account = cursor.fetchone()
            if account != 0:
                return jsonify(status=401, msg="Ya existe este Usuario...Vuelva a intentarlo!")
            else:
                cursor.callproc('insert_newusr', [username, encryptpass, name, lastname, telephone, identif, user_uuid])
                connection.commit()
                return jsonify(status= 200, msg="Datos de usuario registrado con éxito!")

        except Error as error:
            print(error)
        finally:
            connection.close()


# ---------------------------------------------------Update User Record Request---------------------------------------------------

@app.route('/updt_user', methods=['POST'])
#@req_token
def updt_user_info():
    if request.method == 'POST':

        usr_id = request.form['iduser']
        usr_mail = request.form['username']
        usr_pass = request.form['password']
        usr_name = request.form['name']
        usr_last = request.form['lastname']
        usr_tel = request.form['telephone']
        usr_ident = request.form['identification']
        user_uuid = str(uuid.uuid4())

        encryptpass = bcrypt.generate_password_hash(usr_pass).decode('utf-8')

        try:
            connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
            cursor = connection.cursor()
            cursor.callproc()
            chk_exist_data = cursor.fetchone('',[])
            if chk_exist_data != 0:
                return jsonify(status= 304, msg="Los datos requieren ser modificados al menos un valor!")
            else:
                cursor.callproc('update_usr', [usr_mail, encryptpass, usr_name, usr_last, usr_tel, usr_ident, usr_id, user_uuid])
                connection.commit()
                return jsonify(status= 200, msg="Datos de perfil de usuario actualizado!")

        except Error as error:
            print(error)
            #msgfail = "Error al Actualizar Perfil de Usuario Actualizado!"
            #return jsonify(status= ,msg=msgfail)
        finally:
            connection.close()


# ---------------------------------------------------List User Request---------------------------------------------------

@app.route('/list_user', methods=['GET'])
def list_user_info():
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'GET':

        # userid = request.form['id_user']
        userid = request.form.get('id_user')

        try:
            connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
            cursor = connection.cursor()

            # -> Replaced cursor Query for store procedure to improve security at moment fo bringing datas
            cursor.callproc('get_usrdata', [userid])
            row = cursor.fetchone()

            user_mail = row[0]
            user_name = row[1]
            user_last = row[2]
            user_tel = row[3]
            user_ident = row[4]

            return jsonify(mail=user_mail, name=user_name, lastn=user_last, telephone=user_tel, ident=user_ident)

        except Error as error:
            print(error)
        finally:
            cursor.close()
            connection.close()


# ---------------------------------------------------List User Monitorin Position Function---------------------------------------------------

def get_user_position(date):
    markers = list()
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('get_monitlist', [date])
        datas = cursor.fetchall()

        for r in datas:
            marker_dict = dict(city=str(r[0]), date=str(r[1]), latitude=str(r[2]), longitude=str(r[3]),
                               people=str(r[4]), plague=str(r[5]))
            markers.append(marker_dict)

    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()
    print(markers)
    return markers


# ---------------------------------------------------List User Hisotiry Function---------------------------------------------------

def get_position(date, usu):
    markers = list()
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('get_usrhist', [date, usu])
        datas = cursor.fetchall()

        for r in datas:
            marker_dict = dict(latitude=str(r[0]), longitude=str(r[1]), plague=str(r[2]), city=str(r[3]), id=int(r[4]))
            markers.append(marker_dict)

    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    print(markers)
    return markers


# ---------------------------------------------------List Pest Life Cycle Function---------------------------------------------------

def get_life_cycle(pest_values):
    life_cycle = list()
    chain_received = len(pest_values.split(','))
    print("size of chain:", chain_received)

    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()

        if (chain_received > 1):
            print("There's more than one object...")
            my_list = pest_values.split(",")
            # removing any kind of duplicates pest values in the lists with set() function
            lista = list(set(my_list))
            print(my_list)
            cursor.callproc('get_mlifecycl', [lista])

        elif (chain_received == 1):
            print("There's only one object...")
            cursor.callproc('get_slifecycl', [pest_values])
        datas = cursor.fetchall()

        for r in datas:
            row = dict(name=str(r[0]), life_cycle=str(r[1]), population=str(r[2]))
            life_cycle.append(row)

    except Error as error:
        print(error)
    finally:
        cursor.close()
        connection.close()

    return life_cycle


# ---------------------------------------------------Image Prediction Function---------------------------------------------------

def predict_image(request_image, image_name):
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
    lista_pred = []
    lista_predfinal = []

    print('detections:')
    # loop to remove trash prediction values
    for i in range(nums[0]):
        pred_ardict = dict(name=str(class_names[int(classes[0][i])]), scores=float(np.array(scores[0][i])))
        lista_pred.append(pred_ardict)

    for key, group in groupby(lista_pred, lambda x: x["name"]):
        max_y = 0
        for res in group:
            max_y = max(max_y, res["scores"])
        lista_predfinal.append({"name": key, "scores": max_y})
    save_history(lista_predfinal)

    print('deployable list results to DB:', lista_predfinal)

    # test = dict((k, v) for k, v in res.items() if v >= 5)
    # print(test)

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


# ---------------------------------------------------Image Post Request---------------------------------------------------

@app.route('/image', methods=['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    if image_name.endswith('.png'):
        image.save(output_path + image_name)

        im = Image.open(output_path + image_name)
        rgb_im = im.convert('RGB')
        last = image_name[:4]
        nuevo = last + ".jpg"
        rgb_im.save(output_path + nuevo)

        os.remove(output_path + image_name)

        requ = Image.open(output_path + nuevo)
        lt = nuevo
        df = predict_image(requ, lt)

    else:
        request_image = Image.open(image)
        df = predict_image(request_image, image_name)

    try:
        return Response(response=df, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)


@app.route('/monitoring_list_map', methods=['GET'])
@req_token
def monitoring_listmap():
    if request.method == 'GET':
        print("Listing Monitored Users...")
        date = request.form.get('date')
        return jsonify(get_user_position(date))


@app.route('/list_map', methods=['GET'])
def pest_listmap():
    if request.method == 'GET':
        print("Listing User History....")
        date = request.form.get('date')
        usu = request.form.get('user')
        return jsonify(get_position(date, usu))


#@limiter.limit("3 per minute", key_func = lambda : check_user(request.form['username']) != True and request.form['ipadress'])
@app.route('/uplad_file', methods=['POST'])
@req_token
def upl_file_dbx(current_user):
    if request.method == 'POST':
            ip = request.form['hola']
            #ip = request.headers['x-access-token']
            #valor_final = json.dumps(chk_ipornode_one(ip),ensure_ascii=False)
            #data = jwt.decode(ip, os.getenv("app_key"), algorithms='HS256')
            currentDate = datetime.utcnow() +timedelta(seconds=15)
            print(ip)
            #print(data.get('exp'))
            return jsonify(value= current_user)

    """
        print("Preparing upload file....")
        up_file = request.files['image']
        image_name = up_file.filename
        sec_file = secure_filename(image_name)
        dbx_upload(up_file.read(), sec_file)
        return jsonify("hola")"""



# Run server
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
