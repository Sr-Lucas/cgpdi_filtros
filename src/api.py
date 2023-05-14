import os
from PIL import Image
from flask import Flask, jsonify, request, flash, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

import math
from scipy import stats as st
import numpy as np
from matplotlib import pyplot as plt
import cv2

from time import time


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

FLASK_HOST = 'localhost'
FLASK_PORT = '3333'


def create_app(environ, start_response):
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    CORS(app, resources={r"/*": {"origins": "*"}})

    def allowedFile(filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/upload_file', methods=['POST'])
    def uploadFile():
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return jsonify({ "success": False})
        
        file = request.files['file']
        
        ext = file.filename.split('.')
        filename = str(time()).replace('.', '') + "." + ext[1]

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return jsonify({ "success": True, "data": { "filename": filename }})

    @app.route('/files/<filename>', methods=['GET'])
    def getFile(filename):
        return send_file(f"../filtered/{filename}")

    @app.route('/files/hist/<filename>', methods=['GET'])
    def getHistFile(filename):
        return send_file(f"../hist/{filename}")

    ### ROUTES
    @app.route('/negative_filter', methods=['POST'])
    def negativeFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })

        img = Image.open("./uploads/" + filename)
        rimg = filter_core.negativeFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/logarithmic_filter', methods=['POST'])
    def logaritmicFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.logFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/inverse_logaritmic_filter', methods=['POST'])
    def inverseLogaritmicFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.inverseLogFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/nth_power_filter', methods=['POST'])
    def nthPoewerFilter():
        data = request.get_json()
        filename = data.get('filename')
        gamma = data.get('gamma')

        print(filename)
        print(gamma)

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.nthPoewerFilter(img, gamma)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/nth-root-filter', methods=['POST'])
    def nthRootFilter():
        data = request.get_json()
        filename = data.get('filename')
        gamma = data.get('gamma')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.nthRootFilter(img, gamma)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/horizontal-mirror-filter', methods=['POST'])
    def horizontalMirrorFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.horizontalMirroringFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/vertical-mirror-filter', methods=['POST'])
    def verticalMirrorFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.verticalMirroringFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/rotation-90-clockwise-filter', methods=['POST'])
    def rotation90ClockwiseFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.rotation90clockwise(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/rotation-90-anticlockwise-filter', methods=['POST'])
    def rotation90AnticlockwiseFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.rotation90anticlockwise(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/rotation-180-filter', methods=['POST'])
    def rotation180Filter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.rotation180(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/compression', methods=['POST'])
    def compression():
        data = request.get_json()
        filename = data.get('filename')
        a = data.get('a')
        b = data.get('b')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.compression(img, a, b)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/expansion', methods=['POST'])
    def expansion():
        data = request.get_json()
        filename = data.get('filename')
        a = data.get('a')
        b = data.get('b')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.expansion(img, int(a), int(b))

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/max-filter', methods=['POST'])
    def maxFilter():
        print('teste')
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.maxFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/min-filter', methods=['POST'])
    def minFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.minFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/moda-filter', methods=['POST'])
    def modaFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.modaFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/pseudo-mediana-filter', methods=['POST'])
    def pseudoMedianaFilter():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.pseudoMedianaFilter(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/k-nearest-neighbour', methods=['POST'])
    def kNearestNeighbour():
        data = request.get_json()
        filename = data.get('filename')
        k = data.get('k')

        print(filename, k)

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.kNearestNeightborFilter(img, int(k))

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/make-histogram', methods=['POST'])
    def makeHistogram():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        filename = filter_core.makeImghistogram(img)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/hist/{filename}"
            } 
        })

    @app.route('/equalize-image', methods=['POST'])
    def equalizeImage():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.equalizateImage(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/sum-images', methods=['POST'])
    def sumImages():
        data = request.get_json()
        filename1 = data.get('filename1')
        filename2 = data.get('filename2')

        if not filename1 or not filename2:
            return jsonify({ "success": False })
        
        filename = filter_core.sumImages("./uploads/" + filename1, "./uploads/" + filename2)


        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
            } 
        })

    @app.route('/laplaciano', methods=['POST'])
    def laplaciano():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.laplaciano(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/hightboost', methods=['POST'])
    def hightboost():
        data = request.get_json()
        filename = data.get('filename')
        p = data.get('p')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.hightBoost(img, int(p))

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/prewitt', methods=['POST'])
    def prewitt():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.prewitt(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/sobel', methods=['POST'])
    def sobel():
        data = request.get_json()
        filename = data.get('filename')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.sobel(img)

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })


    @app.route('/simulate_grey_level_reduction', methods=['POST'])
    def simulate_grey_level_reduction():
        data = request.get_json()
        filename = data.get('filename')
        n = data.get('n')

        if not filename:
            return jsonify({ "success": False })
        
        img = Image.open("./uploads/" + filename)
        rimg = filter_core.simulateGrayLevelPalletRedution(img, int(n))

        ext = filename.split('.')[1]
        newFilename = str(time()).replace('.', '') + "." + ext

        rimg.save('./filtered/' + newFilename)

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/amplify_neartest_neightbor', methods=['POST'])
    def amplify_neartest_neightbor():
        data = request.get_json()
        filename = data.get('filename')
        scale_factor = data.get('scale_factor')

        if not filename:
            return jsonify({ "success": False })
        
        newFilename = filter_core.expand_image_nn("./uploads/" + filename, int(scale_factor))

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    @app.route('/amplify_bilinear', methods=['POST'])
    def amplify_bilinear():
        data = request.get_json()
        filename = data.get('filename')
        scale_factor = data.get('scale_factor')

        if not filename:
            return jsonify({ "success": False })
        
        newFilename = filter_core.expand_image_bilinear("./uploads/" + filename, int(scale_factor))

        return jsonify({ 
            'success': True, 
            "data": {
                'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{newFilename}"
            } 
        })

    return app











## ## ## ## ##





import math
from time import time
from scipy import stats as st
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

MASK_SIZE = 3

def negativeFilter(img):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):        
        pixel = img.getpixel((i,j))

        outputPixel = 255 - pixel
        copy.putpixel((i,j), outputPixel)
  
  return copy



def logFilter(img):
  copy = img.copy()

  c = 255/math.log(255+1, 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(logTransform(c, pixel))
      copy.putpixel((i,j),(outputPixel))
  
  return copy

def logTransform(c, pixel):
    return c * math.log(float(1 + pixel), 10)



def inverseLogFilter(img):
  copy = img.copy()

  c = 255/((255+1) ** 10)

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      # Get pixel value at (x,y) position of the image
      pixel = img.getpixel((i,j))

      # Do log transformation of the pixel
      outputPixel = round(invLogTransform(c, pixel))
      copy.putpixel((i,j),(outputPixel))
  
  return copy

def invLogTransform(c, pixel):
  return c * (float(1 + pixel) ** 10)


def nthPoewerFilter(img, gamma = 1.0):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      pixel = img.getpixel((i,j))

      outputPixel: float = 255 * ((pixel / 255) ** gamma)
      copy.putpixel((i,j),(outputPixel.__round__()))
  
  return copy



def nthRootFilter(img, gamma = 1.0):
  copy = img.copy()

  for i in range(0, img.size[0]-1):
    for j in range(0, img.size[1]-1):
      pixel = img.getpixel((i,j))

      outputPixel = 255 * ((pixel / 255) ** (1/gamma))
      copy.putpixel((i,j),(outputPixel.__round__()))
  
  return copy


def horizontalMirroringFilter(img: Image):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - x, y)))
  
  return imgCopy


def verticalMirroringFilter(img: Image):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel((x, (img.size[0]-1) - y)))
  
  return imgCopy

# G(x, y) = F(y, (TAM - 1) - x)
def rotation90clockwise(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel((y, (img.size[0]-1) - x)))
  
  return imgCopy

# G(x, y) = F((TAM - 1) - y, x)
def rotation90anticlockwise(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - y, x)))
  
  return imgCopy

# G(x, y) = F((TAM - 1) - x, (TAM - 1) - y)
def rotation180(img):
  imgCopy = img.copy()
  if (img.size[0] != img.size[1]):
    return 0

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      imgCopy.putpixel((x, y), img.getpixel(((img.size[0]-1) - x, (img.size[0]-1) - y)))
  
  return imgCopy



def compression(img, a = 3, b = 1):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      pixel = img.getpixel((x, y))
      rPixel = (pixel / a) - b
      imgCopy.putpixel((x, y), rPixel.__round__())
  
  return imgCopy


def expansion(img, a = 3, b = 1):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      pixel = img.getpixel((x, y))
      rPixel = (a * pixel) - b
      imgCopy.putpixel((x, y), rPixel.__round__())
  
  return imgCopy

def fillMask(img, x, y):
  mask = np.zeros((3, 3))

  mask[0][0] = img.getpixel((x-1, y-1))
  mask[0][1] = img.getpixel((x, y-1))
  mask[0][2] = img.getpixel((x+1, y-1))
  mask[1][0] = img.getpixel((x-1, y))
  mask[1][1] = img.getpixel((x, y))
  mask[1][2] = img.getpixel((x+1, y))
  mask[2][0] = img.getpixel((x-1, y+1))
  mask[2][1] = img.getpixel((x, y+1))
  mask[2][2] = img.getpixel((x+1, y+1))
  
  return mask


def maxFilter(img):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      max = 0
      for a in range(0, 3):
        for b in range(0, 3):
          if (mask[a][b] > max):
            max = mask[a][b]
      
      imgCopy.putpixel((x, y), max.__round__())
  
  return imgCopy


def minFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      min = 255
      for a in range(0, 3):
        for b in range(0, 3):
          if (mask[a][b] < min):
            min = mask[a][b]
      
      imgCopy.putpixel((x, y), min.__round__())
  
  return imgCopy



def modaFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      modeResult = st.mode(mask.flatten())

      imgCopy.putpixel((x, y), modeResult.mode[0].__round__())
  
  return imgCopy


def pseudoMedianaFilter(img):
  imgCopy = img.copy()
  mask = np.zeros((3, 3))

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)
      L = MASK_SIZE * MASK_SIZE
      M = int((L + 1)/2)

      maskFlattened = mask.flatten()

      arr = []
      i = 0
      while M+i <= L:
        arr.append([])
        for k in range(i, M+i):
          arr[i].append(maskFlattened[k])
        i += 1
      
      maxes = []
      mins = []
      for i in range(0, arr.__len__()-1):
        maxes.append(max(arr[i]))
        mins.append(min(arr[i]))
      
      minmax = min(maxes)
      maxmin = max(mins)

      pseudoMedian = (maxmin + minmax)/2

      imgCopy.putpixel((x, y), pseudoMedian.__round__())
  
  return imgCopy


def expand_image_nn(image_path, scale_factor):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rows = image.shape[0]
    cols = image.shape[1]

    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    result = np.zeros((new_rows, new_cols), dtype=image.dtype)

    row_scale = float(rows) / new_rows 
    col_scale = float(cols) / new_cols

    for i in range(new_rows):
        for j in range(new_cols):
            row = int(i * row_scale) # 0 0 1 1 2 2 3 3
            col = int(j * col_scale) # 0 0 1 1 2 2 3 3
            result[i, j] = image[row, col]
    
    ext = image_path.split('/').pop().split('.')[1]
    filename = str(time()).replace('.', '') + "." + ext

    cv2.imwrite(f"./filtered/{filename}", result)

    return filename

def expand_image_bilinear(input_path, scale_factor):
   # Load an image using OpenCV
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Get the size of the original image
    rows = image.shape[0]
    cols = image.shape[1]
    
    # Calculate the new size of the image
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)
    
    # Create a blank canvas with the new size
    result = np.zeros((new_rows, new_cols), dtype=image.dtype)
    
    # Calculate the scale factors
    row_scale = float(rows - 1) / (new_rows - 1) # 255/511 = 0.5
    col_scale = float(cols - 1) / (new_cols - 1) # 255/511 = 0.5

    # Iterate over the new image and find the values using bilinear interpolation
    for i in range(new_rows): # 0 1 2 3 4 5 6 7
        for j in range(new_cols): # 0 1 2 3 4 5 6 7
            row = i * row_scale # 0 0.5 1 1.5 2 2.5 3 3.5  
            col = j * col_scale # 0 0.5 1 1.5 2 2.5 3 3.5

            row_1 = int(row) # 0 0 1 1 2 2 3 3
            row_2 = min(int(row) + 1, rows - 1) #
            col_1 = int(col) 
            col_2 = min(int(col) + 1, cols - 1)

            value_1 = (col_2 - col) * image[row_1, col_1] + (col - col_1) * image[row_1, col_2]
            value_2 = (col_2 - col) * image[row_2, col_1] + (col - col_1) * image[row_2, col_2]

            result[i, j] = (row_2 - row) * value_1 + (row - row_1) * value_2

    ext = input_path.split('/').pop().split('.')[1]
    filename = str(time()).replace('.', '') + "." + ext

    cv2.imwrite(f"./filtered/{filename}", result)

    return filename


def kNearestNeightborFilter(img, k: int):
  kn = k
  if kn >= (MASK_SIZE * MASK_SIZE) - 1:
    kn = (MASK_SIZE * MASK_SIZE) - 1

  imgCopy = img.copy()
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      middleX = (MASK_SIZE/2).__floor__()
      middleY = (MASK_SIZE/2).__floor__()

      pivo = mask[middleX][middleY]
    
      maskFlatten = mask.flatten()
      maskFlatten[int(mask.size/2)] = 99999999

      pivoDistanceArr = []
      for i in range(0, mask.size):
        pivoDistanceArr.append(abs(maskFlatten[i] - pivo))
      
      sortedArr = pivoDistanceArr.copy()
      sortedArr.sort()

      sumRes = 0
      for i in range(0, k):
        index = pivoDistanceArr.index(sortedArr[i])
        sumRes += maskFlatten[index]
      
      knnResult = sumRes/k

      imgCopy.putpixel((x, y), knnResult.__round__())

  return imgCopy


def makeImghistogram(img):  
  ih = img.size[0]-1
  iw = img.size[1]-1

  hist = np.zeros([256], np.int32)

  for x in range(0, ih):
    for y in range(0, iw):
      hist[img.getpixel((y, x))] += 1
  
  # make an image name of a timestamp
  filename = str(time()).replace('.', '') + "." + 'jpg'
  
  plt.figure()
  plt.title('Gray Scale Histogram')
  plt.xlabel('Intensity Level')
  plt.ylabel('Intesity Frequency')
  plt.xlim([0, 256])
  plt.plot(hist)
  plt.savefig(f"./hist/{filename}")

  return filename


def equalizateImage(img):
  imgArr = np.asarray(img)
  hist = np.bincount(imgArr.flatten(), minlength=256)

  numPixels = np.sum(hist)
  hist = hist/numPixels

  chistogramArr = np.cumsum(hist)

  transforMap = np.floor(255 * chistogramArr).astype(np.uint8)

  imgList = list(imgArr.flatten())

  eqImgList = [transforMap[p] for p in imgList]

  eqImgArr = np.reshape(np.asarray(eqImgList), imgArr.shape)

  equalizedImage = Image.fromarray(eqImgArr, mode='L')

  return equalizedImage


def sumImages(img1Path: str, img2Path: str, img1percentual = 0.5, img2percentual = 0.5):
  img1 = cv2.imread(img1Path)
  img2 = cv2.imread(img2Path)
  dst = cv2.addWeighted(img1, img1percentual, img2, img2percentual, 0)

  ext = img1Path.split('/').pop().split('.')[1]
  filename = str(time()).replace('.', '') + "." + ext

  cv2.imwrite(f"./filtered/{filename}", dst)

  return filename


def laplaciano(img):
  # 2x NC central - NC Esquerdo - NC Direito

  imgCopy = img.copy()
    
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      z1 = mask[0][0]
      z2 = mask[0][1]
      z3 = mask[0][2]
      z4 = mask[1][0]
      z5 = mask[1][1]
      z6 = mask[1][2]
      z7 = mask[2][0]
      z8 = mask[2][1]
      z9 = mask[2][2]

      pixelR = (- (z1 + z2 + z3 + z4 + z6 + z7 + z8 + z9)) + (8 * z5)

      if pixelR > 255:
        pixelR = 255
      elif pixelR < 0:
        pixelR = 0

      imgCopy.putpixel((x, y), int(pixelR))
    
  return imgCopy


def hightBoost(img, percentual: float):
  A = 1 + percentual

  imgCopy = img.copy()
    
  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      z1 = mask[0][0]
      z2 = mask[0][1]
      z3 = mask[0][2]
      z4 = mask[1][0]
      z5 = mask[1][1]
      z6 = mask[1][2]
      z7 = mask[2][0]
      z8 = mask[2][1]
      z9 = mask[2][2]

      pixelR = (- (z1 + z2 + z3 + z4 + z6 + z7 + z8 + z9)) + (8 * z5)
      pixelR = pixelR + (A * z5)


      if pixelR > 255:
        pixelR = 255
      elif pixelR < 0:
        pixelR = 0

      imgCopy.putpixel((x, y), int(pixelR))
    
  return imgCopy


def prewitt(img):
    imgCopy = img.copy()

    for y in range(0, img.size[0]-1):
      for x in range(0, img.size[1]-1):
        mask = fillMask(img, x, y)

        z1 = mask[0][0]
        z2 = mask[0][1]
        z3 = mask[0][2]
        z4 = mask[1][0]
        z5 = mask[1][1]
        z6 = mask[1][2]
        z7 = mask[2][0]
        z8 = mask[2][1]
        z9 = mask[2][2]

        pixelR = abs((z7 + z8 + z9) - (z1 + z2 + z3)) + abs((z3 + z6 + z9) - (z1 + z4 + z7))

        if pixelR > 255:
          pixelR = 255
        elif pixelR < 0:
          pixelR = 0

        imgCopy.putpixel((x, y), int(pixelR))
    

    return imgCopy



def sobel(img):
  imgCopy = img.copy()

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      mask = fillMask(img, x, y)

      z1 = mask[0][0]
      z2 = mask[0][1]
      z3 = mask[0][2]
      z4 = mask[1][0]
      z5 = mask[1][1]
      z6 = mask[1][2]
      z7 = mask[2][0]
      z8 = mask[2][1]
      z9 = mask[2][2]

      pixelR = abs((z7 + (2*z8) + z9) - (z1 + (2*z2) + z3)) + abs((z3 + (2*z6) + z9) - (z1 + (2*z4) + z7))
      
      if pixelR > 255:
        pixelR = 255
      elif pixelR < 0:
        pixelR = 0

      imgCopy.putpixel((x, y), int(pixelR))
    
  return imgCopy


def simulateGrayLevelPalletRedution(img, n):
  imgCopy = img.copy()

  # first, get the image histogram
  ih = img.size[0]-1
  iw = img.size[1]-1

  hist = np.zeros([256], np.int32)

  for x in range(0, ih):
    for y in range(0, iw):
      hist[img.getpixel((y, x))] += 1
  
  # get the n most frequent colors
  mostFrequentColors = np.argsort(hist)[::-1][:n]

  for y in range(0, img.size[0]-1):
    for x in range(0, img.size[1]-1):
      pixel = img.getpixel((x, y))

      # if the pixel is one of the most frequent colors, keep it
      if pixel in mostFrequentColors:
        imgCopy.putpixel((x, y), int(pixel))
        continue;

      # get the closest color
      closestColor = mostFrequentColors[0]
      for color in mostFrequentColors:
        if abs(pixel - color) < abs(pixel - closestColor):
          closestColor = color

      imgCopy.putpixel((x, y), int(closestColor))
    
  return imgCopy
