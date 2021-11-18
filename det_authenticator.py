from functools import wraps
from flask import request, jsonify
from datetime import datetime, timedelta
from multiprocessing import Value
from psycopg2 import Error
from dotenv import load_dotenv
import requests, json
import psycopg2 as psy
import shodan
import jwt
import os
import re
counter = Value('i', 0)


host_con = os.getenv("host_con")
port_con = os.getenv("port_con")
db_con = os.getenv("db_con")
user_con = os.getenv("user_con")
pass_con = os.getenv("pass_con")

load_dotenv()
#------------------------------Creating a class to set and get values------------------------------
class count_fail:

    def __init__(self, pnlt_time):
        self.pnlt_time= pnlt_time

    def get_pnlt_time(self):
        return self.pnlt_time

    def set_pnlt_time(self, pnlt_time):
        self.pnlt_time = pnlt_time


def count_status(status):
    if not status == True:
        counter.value += 1
        cout_store = counter.value
    else:
        counter.value = 1
        cout_store = counter.value

    return cout_store


def chk_user_penalty(usr_ipadd,conn):
    try:
        cursor = conn.cursor()
        cursor.callproc('get_penalties', [usr_ipadd])
        row = cursor.fetchone()
        if row:
            print(row[0])
            return True
        else:
            return False
    except Error as error:
        print(error)


def sv_user_penalty(usr_ipadd,date_pn,conn):
    try:
        cursor = conn.cursor()
        cursor.callproc('save_penalty', [usr_ipadd,date_pn])
        conn.commit()
    except Error as error:
        print(error)

#---------------------------------IP, VPN or Thor detector---------------------------------
def chk_ipornode_one(ip_hola):
    try:
        ip_check = requests.get("https://vpnapi.io/api/"+ ip_hola + "?key=" + os.getenv("ip_validator"))
        data = json.loads(ip_check.text)

        if data.get("security") is None:
            return {"status":False, "msg":"None"}
        elif data.get("security").get("vpn"):
            return {"status":True,"msg": f"Queremos ver su verdadera identidad, la direccion {ip_hola} es una VPN y no la admitimos por razones de seguridad!"}
        elif data.get("security").get("tor"):
            return {"status":True,"msg": f"Queremos ver su verdadera identidad, la direccion {ip_hola} es un nodo de TOR y no la admitimos por razones de seguridad!"}
        elif data.get("security").get("proxy"):
            return {"status":True,"msg": f"Queremos ver su verdadera identidad, la direccion {ip_hola} es una Proxy y no la admitimos por razones de seguridad!"}
    except Error as err:
            print(err)



def chk_ipornode_two():
    try:
        print()
    except Error as err2:
        print(err2)




#------------------------------JWT builder and token validator------------------------------
def auth_builder(uui_usr):
    token = jwt.encode({'public_id': uui_usr, 'exp': datetime.utcnow() + timedelta(minutes=1)},os.getenv("app_key"))
    return token


#-------------------Required token function to check if user is valid or not------------------
def req_token(f):
    @wraps(f)
    def check_token(*args, **kwargs):
        token_payload = None

        if 'access_token' in request.headers:
            token_payload = request.headers['access_token']
        if not token_payload:
            return jsonify(status= 401, msg= "El sistema de autenticacion requiere de un Token!")
        try:
            data = jwt.decode(token_payload, os.getenv("app_key"), algorithms='HS256')
            connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
            cursor = connection.cursor()
            cursor.callproc('get_loguuid', [data.get('public_id')])
            current_user = cursor.fetchone()
            print(current_user)
        except jwt.ExpiredSignatureError as jwser:
            print(jwser)
            return jsonify(status= 440, msg= "Error...Sesión Expirada!")
        except jwt.InvalidTokenError as jwie:
            print(jwie)
            return jsonify(status= 403, msg= "Error...Token Invalido!")
        #return current_user in dicts
        return f(current_user, *args, **kwargs)
    return check_token


#----------------------------------BlackList for expired token----------------------------------
def blist_oken(jwt_payload):
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('', [jwt_payload])
        token_black_list = cursor.fetchone()
        return token_black_list is not None
    except Error as err:
        print("Error al verif. expiracion de Token:",err)


def reg_blist_token(jwt_payload):
    try:
        connection = psy.connect(host=host_con, port=port_con, database=db_con, user=user_con, password=pass_con)
        cursor = connection.cursor()
        cursor.callproc('', [jwt_payload])
        connection.commit()
        print("Token registrado!")
    except Error as err:
        print("Error al intentar almacenar el token expirado:",err)
    finally:
        cursor.close()
        connection.close()












