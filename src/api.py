import os
from PIL import Image
from flask import Flask, jsonify, request, flash, send_file
from werkzeug.utils import secure_filename

from time import time

import filter_core


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

FLASK_HOST = 'localhost'
FLASK_PORT = '3333'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['POST'])
def uploadFile():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
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
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/logaritmic_filter')
def logaritmicFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.logFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/inverse_logaritmic_filter')
def inverseLogaritmicFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.inverseLogFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/nth-power-filter')
def nthPoewerFilter():
    data = request.get_json()
    filename = data.get('filename')
    gamma = data.get('gamma')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.nthPoewerFilter(img, gamma)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/nth-root-filter')
def nthRootFilter():
    data = request.get_json()
    filename = data.get('filename')
    gamma = data.get('gamma')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.nthRootFilter(img, gamma)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/horizontal-mirror-filter')
def horizontalMirrorFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.horizontalMirroringFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/vertical-mirror-filter')
def verticalMirrorFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.verticalMirroringFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/rotation-90-clockwise-filter')
def rotation90ClockwiseFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.rotation90clockwise(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/rotation-90-anticlockwise-filter')
def rotation90AnticlockwiseFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.rotation90anticlockwise(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/rotation-180-filter')
def rotation180Filter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.rotation180(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/compression')
def compression():
    data = request.get_json()
    filename = data.get('filename')
    a = data.get('a')
    b = data.get('b')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.compression(img, a, b)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/expansion')
def expansion():
    data = request.get_json()
    filename = data.get('filename')
    a = data.get('a')
    b = data.get('b')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.expansion(img, a, b)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/max-filter')
def maxFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.maxFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/min-filter')
def minFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.minFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/moda-filter')
def modaFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.modaFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/pseudo-mediana-filter')
def pseudoMedianaFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.pseudoMedianaFilter(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/nnr-ampliation')
def nnrAmpliation():
    data = request.get_json()
    filename = data.get('filename')
    size = data.get('size')

    # size has to be 256 or 512 or 1024
    if size not in [256, 512, 1024]:
        return jsonify({ "success": False })

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.nnrAmpliation(img, int(size))
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/bir-ampliation')
def birAmpliation():
    data = request.get_json()
    filename = data.get('filename')
    size = data.get('size')

    # size has to be 256 or 512 or 1024
    if size not in [256, 512, 1024]:
        return jsonify({ "success": False })

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.birAmpliation(img, int(size))
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/k-nearest-neighbour')
def kNearestNeighbour():
    data = request.get_json()
    filename = data.get('filename')
    k = data.get('k')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.kNearestNeighbour(img, int(k))
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/make-histogram')
def makeHistogram():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    filename = filter_core.makeHistogram(img)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/hist/{filename}"
        } 
    })

@app.route('/equalize-image')
def equalizeImage():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.equalizeImage(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/sum-images')
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

@app.route('/laplaciano')
def laplaciano():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.laplaciano(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/hightboost')
def hightboost():
    data = request.get_json()
    filename = data.get('filename')
    k = data.get('k')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.hightBoost(img, int(k))
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/prewitt')
def prewitt():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.prewitt(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

@app.route('/sobel')
def sobel():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })
    
    img = Image.open("./uploads/" + filename)
    rimg = filter_core.sobel(img)
    rimg.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

app.run(port=FLASK_PORT,host=FLASK_HOST,debug=True)
