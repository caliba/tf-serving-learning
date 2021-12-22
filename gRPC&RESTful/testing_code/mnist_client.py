import pickle
import numpy as np
import requests

host = 'localhost'
port = '8501'
batch_size = 1
image_path = "./mnist_image.pkl"
model_name = 'mnist'
signature_name = 'predict_images'

with open(image_path, 'rb') as f:
    image = pickle.load(f)
batch = np.repeat(image, batch_size, axis=0).tolist()
json = {
    "signature_name": signature_name,
    "instances": batch
}
response = requests.post("http://%s:%s/v1/models/mnist:predict" % (host, port), json=json)
