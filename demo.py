import numpy as np
import cv2
import json
import os,sys
from tritonclient.utils import *
import tritonclient.http as httpclient
import random

DEBUG = bool(int(os.environ.get('DEBUG', 0)))
#MODEL_URL = os.environ.get('MODEL_URL', "172.16.30.4:4000")
MODEL_URL = os.environ.get('MODEL_URL', "172.18.128.120:8000")
MODEL_NAME = os.environ.get('MODEL_NAME', "text_server_table_pp")
#MODEL_NAME = os.environ.get('MODEL_NAME', "dev_ensemble")


def get_files(filepath, exts):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} files'.format(len(files)))
    files = sorted(files)
    return files
    
def make_input_jpeg(img_bytes):
    img_binary = np.array([img_bytes], dtype = np.object_)
    img_json_raw = {
      'client_id': 'example', 
      'params': [
        {
          'id': '0',
          'type': 'jpg',
          'data': 'IMAGE_BINARY'
        }
      ]  
    }
    img_json = np.array([json.dumps(img_json_raw)], dtype = np.object_)
    return img_binary, img_json
  

def make_input_raw(img_bytes):
    img_mat = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) # NHWC
    img_binary = np.array([img_mat.tobytes()], dtype = np.object_)
    img_shape = list(img_mat.shape)
    img_json_raw = {
      'client_id': 'example', 
      'params': [
        {
          'id': '0',
          'type': 'raw',
          'data': 'IMAGE_BINARY',
          'shape': img_shape
        }
      ]  
    }

    img_json = np.array([json.dumps(img_json_raw)], dtype = np.object_)
    return img_binary, img_json
    
    
def predict_by_client(filename):
    verbose = False
    #filename = 'tmp.jpg'
    #cv2.imwrite(filename, image)
        
    bbox, classes, scores = [], [], []
        
    with httpclient.InferenceServerClient(MODEL_URL, verbose=verbose) as client:
        model_metadata: dict = client.get_model_metadata(model_name=MODEL_NAME)
        with open (filename, 'rb') as img_file:
            img_bytes = img_file.read()
            ext = os.path.splitext(os.path.basename(filename))[-1]
            
            if ext == 'jpg' or ext == 'jpeg':
                img_binary, img_json = make_input_jpeg(img_bytes)
            else:
                img_binary, img_json = make_input_raw(img_bytes)
                
            inputs = [
                  httpclient.InferInput('IMAGE_BINARY', img_binary.shape, np_to_triton_dtype(img_binary.dtype)),
                  httpclient.InferInput('IMAGE_DESC', img_json.shape, np_to_triton_dtype(img_json.dtype))
                  ]
            inputs[0].set_data_from_numpy(img_binary)
            inputs[1].set_data_from_numpy(img_json)
            
            outputs = [httpclient.InferRequestedOutput("RESULT")]
            #结果解析
            response = client.infer(MODEL_NAME, inputs, request_id=str(1), outputs=outputs)
            out_json = response.as_numpy('RESULT')
            out_str = out_json[0].decode()
            out_str = json.loads(out_str)
             
            #print(out_str)
    return out_str    
            
def predict_image_table():
    filepath = '/mnt/disk2/wyc/ocr/data/jpg'
    savepath = '/mnt/disk2/wyc/ocr/output'
    files = get_files(filepath, ['jpg', 'png', 'JPG', 'PNG'])
    print(files)
    color = (0,0,255)
    for filename in files:
        print(filename)
        res = predict_by_client(filename)
        results = res["results"]
        imgname = os.path.basename(filename)
        image = cv2.imread(filename)
        for result in results:
            objects = result["objects"]
            for obj in objects:
                #print(obj)
                position = obj["position"]
                attributes = obj["attributes"]
                for cell in obj["objects"]:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    #print(cell)
                    cell_position = cell["position"]
                    attributes = cell["attributes"]
                    
                    for k in range(len(attributes)):
                        key = str(attributes[k]["key"])
                        value = attributes[k]["value"]
                        print(key, value)
                        
                poly_1d = np.array(position).astype(np.int32).reshape((-1))
                poly = poly_1d.reshape(-1, 2)
                cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color, thickness=2)
                
                x1, y1,x2,y2 = position[0][0], position[0][1], position[2][0], position[2][1]
                
        cv2.imwrite('show/%s.jpg'%(os.path.basename(filename)), image)
    
def main():
    predict_image_table()
    
if __name__=='__main__':
    main()
