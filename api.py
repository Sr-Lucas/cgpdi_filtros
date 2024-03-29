import os
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os

from time import time

import filter_core


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

FLASK_HOST = 'localhost'
FLASK_PORT = '5000'

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
    return send_file(f"./filtered/{filename}")

@app.route('/files/hist/<filename>', methods=['GET'])
def getHistFile(filename):
    return send_file(f"./hist/{filename}")

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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/hist/{filename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{filename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
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
            'url': f"https://cg-pdi-api.onrender.com/files/{newFilename}"
        } 
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=FLASK_PORT)