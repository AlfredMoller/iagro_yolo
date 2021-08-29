import sys
import dropbox
from dropbox.exceptions import ApiError, AuthError


dbox_key = '2aPXeDN95_sAAAAAAAAAAcP47wmRizJ1bHJheJHrJJ_dwVOefc3W7lxgL0MbJPGe'
dbx = dropbox.Dropbox(dbox_key)
dropbox_bdir = '/test_dropbox3/'
pc_bdir = '/detections/resize'
pc_fname = 'detection.jpg'
file_path = pc_bdir + pc_fname
name_dir = 'elizabeth_images'


def dbox_metadata():
    list_dir_ext = list()
    for entry in dbx.files_list_folder("").entries:
        list_dir_ext.append(entry.name)
    return list_dir_ext

print('listado de recursos:',dbox_metadata())



def check_ext(mail):
    if '_images' in mail:
        return True
    else:
        return False


#   Uso de la función filter para
#   comprobar emails válidos de una lista.
def dbx_crt_folder():
    valid_folder_ext = filter(check_ext, dbox_metadata())
    if name_dir in list(valid_folder_ext):
       print("existe el repositorio")
    else:
       print("procedemos a crear repo")



def dbx_upload(filestr,f_name):
    msg_upload = None
    try:
        push_path_folder = dropbox_bdir + f_name
        dbx.files_upload(filestr, push_path_folder, mode=dropbox.files.WriteMode("overwrite"))
    except ApiError as err:
        if(err.error.is_path() and
                    err.error.get_path().error.is_insufficient_space()):
            sys.exit("ERROR: No se puede realizar procedimiento de almacenado, hay poco espacio en el repo principal")
        elif err.user_message_text:
            print(err.user_message_text)
            sys.exit()
        else:
            print(err)
            sys.exit()
