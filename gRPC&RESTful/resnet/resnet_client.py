# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet#pretrained-models

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import base64
import io
import json

import numpy as np
from PIL import Image
import requests

# The server URL specifies the endpoint of your server running the ResNet 服务器URL指定运行ResNet的服务器的端点
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'

# The image URL is the location of the image we should send to the server 我们发送的请求
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False


def main():
  # Download the image
  # stream=false，他会立即开始下载文件并存放到内存当中，倘若文件过大就会导致内存不足的情况
  dl_request = requests.get(IMAGE_URL, stream=True) # 从链接下载输入图片
  dl_request.raise_for_status() # 判断连接结果是不是 200(请求成功) 不是则抛出异常

  if MODEL_ACCEPT_JPG: # 如果能接受jpg格式的图片

    # Compose a JSON Predict request (send JPEG image in base64).
    # base64先编码再解码
    jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8') # 编写json预测请求
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes # 输出格式
  else:
    # Compose a JOSN Predict request (send the image tensor). 编写 JSON 预测请求 发送请求张量
    #Image.open 打开图像是PIL类型，
    # io.BytesIO 读写二进制文件
    #request.content() 存的是图像的字节编码
    jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    # tensorflow，numpy的顺序是(batch,h,w,c)： 所以需要进行PIL类型到numpy类型转换，
    # 先归一化，然后增加一个维度axis = 0 指在0位置添加一个数据
    # .tolist 矩阵转化为列表
    jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
    #
    predict_request = json.dumps({'instances': jpeg_rgb}) # 将一个python 数据结构转换为json格式 生成的是字符串


  # Send few requests to warm-up the model.
  #  模型预热
  # _是循环标志，下面的循环并不会用到
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency
  # 发送请求 并且 报告延迟
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds() # 接口响应时间
    prediction = response.json()['predictions'][0] # 获取预测结果

  print('Prediction class: {}, avg latency: {} ms'.format(
      np.argmax(prediction), (total_time * 1000) / num_requests))


if __name__ == '__main__':
  main()
