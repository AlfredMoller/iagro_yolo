import os
import dropbox
from dropbox.exceptions import ApiError, AuthError
from itertools import filterfalse, repeat
from dotenv import load_dotenv

load_dotenv()
dbox_key = os.getenv("dbox_key1")
dbx = dropbox.Dropbox(dbox_key)
dropbox_bdir = '/test_dropbox3/'


# ------------------------------------------- Remote File uploader -------------------------------------------
def dbox_metadata():
    list_dir_ext = list()
    for entry in dbx.files_list_folder("").entries:
        list_dir_ext.append(entry.name)
    return list_dir_ext


def check_ext(file_ext):
    if '_images' in file_ext:
        return True
    else:
        return False


# Checking the extension of folder instance  exist in all metadatas
def dbx_crt_folder(name_dir):
    try:
        valid_folder_ext = filter(check_ext, dbox_metadata())
        if name_dir in list(valid_folder_ext):
            return {"status": 303, "msg": f"El repo con nombre {name_dir} xiste...Intentelo con otro nombre!"}
        else:
            dbx.files_create_folder_v2("/" + name_dir)
            return {"status": 200, "msg": "El repositorio fue creado exitosamente!"}
    except ApiError as err:
        print(err)


def dbx_upload(filestr, f_name):
    try:
        push_path_folder = dropbox_bdir + f_name
        dbx.files_upload(filestr, push_path_folder, mode=dropbox.files.WriteMode("overwrite"))
    except ApiError as err:
        if (err.error.is_path() and err.error.get_path().error.is_insufficient_space()):
            print("No se puede almacenar datos, hay poco espacio en el repo principal")
        elif err.user_message_text:
            print(err.user_message_text)
        else:
            print(err)


# ---------------------------------------- Current machine file uploader ----------------------------------------

def chk_lclmachine(fold_pth):
    try:
        dir_name = os.path.join(os.path.dirname(__file__), "static")
        cur_path = os.path.join(dir_name, fold_pth)
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
            os.chdir(cur_path)
            for sub_folder in ['aracnido_rojo', 'pulgon', 'mosca_blanca']:
                os.mkdir(sub_folder)
            return {"status": 303, "msg": "El repositorio local, fue creado exitosamente!"}
        else:

            return {"status": 303, "msg": f"El repo con nombre {fol_path} existe...Intentelo de nuevo!"}
    except OSError as oserr:
        print("Error al crear el repositorio:", oserr)
