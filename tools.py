from tensorflow.python.client import device_lib
import os
import pickle

def save(obj, name, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name, folder):
    with open(folder + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']
