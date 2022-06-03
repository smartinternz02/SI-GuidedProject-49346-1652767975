
from flask import Flask,render_template,request, Response
import cv2 # opencv library
from tensorflow.keras.models import load_model#to load our trained model
import numpy as np
#import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from camera1 import VideoCamera

app = Flask(__name__,template_folder="templates",static_url_path='/static') # initializing a flask app
# camera=cv2.VideoCapture(0)
model=load_model('model_building_defects_vgg16.h5')
print("Loaded model from disk")

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')
@app.route('/intro', methods=['GET'])
def about():
    return render_template('intro.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload():
        return render_template("upload1.html")
@app.route('/video')
def video():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
      app.run(host="0.0.0.0",port=5001,debug=False)
 
