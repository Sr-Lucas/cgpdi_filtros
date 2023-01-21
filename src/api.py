import os
from PIL import Image
from flask import Flask, jsonify, request, flash, send_file
from werkzeug.utils import secure_filename

from time import time

from filter_core import inverseLogFilter as ilf, logFilter as lf, negativeFilter as nf

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

### ROUTES
@app.route('/negative_filter', methods=['POST'])
def negativeFilter():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({ "success": False })

    img = Image.open("./uploads/" + filename)
    nf(img)
    img.save('./filtered/' + filename)

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
    lf(img)
    img.save('./filtered/' + filename)

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
    ilf(img)
    img.save('./filtered/' + filename)

    return jsonify({ 
        'success': True, 
        "data": {
            'url': f"http://{FLASK_HOST}:{FLASK_PORT}/files/{filename}"
        } 
    })

app.run(port=FLASK_PORT,host=FLASK_HOST,debug=True)