import os
import matlab.engine
import tensorflow as tf
import numpy as np
import os
import glob
import numpy as np
import cv2
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

app = Flask(__name__)
eng = matlab.engine.start_matlab()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    target2 = os.path.join(APP_ROOT, 'out/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    if not os.path.isdir(target2):
            os.mkdir(target2)
    else:
        print("Couldn't create upload directory: {}".format(target2))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)


    image_path=target + filename
    image_size=128
    images = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = images / 255

    x= graph.get_tensor_by_name("x:0") 
    img_size=128
    num_channels=3
    img_size_flat = img_size * img_size * num_channels
    x_batch = images.reshape(1, img_size_flat)
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) ### 2=>number of classes
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    resultado=sess.run(y_pred, feed_dict=feed_dict_testing)

    output=""
    if resultado[0][0]>resultado[0][1]:
        output="short hair"
    elif resultado[0][1]>resultado[0][0]:
        output="long hair"
    else:
        ouput="cannot be recognized"

     ###################################   
    
    red = int(request.form['R'])
    grn = int(request.form['G'])
    blu = int(request.form['B'])
    fn=upload.filename
    tf = eng.ChangeHairColor2(fn,red,grn,blu)
    
    return render_template("complete.html", image_name=filename, output=output)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("out", filename)

@app.route('/camera')
def camera(name=None):
    #return camera
    return render_template("camera.html",name=name)

@app.route('/gallery')
def gallery(name=None):
    return render_template("gallery.html",name=name)
if __name__ == "__main__":
    app.run(port=8080, debug=True)
