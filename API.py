from flask import Flask,request,send_file
from flasgger import Swagger
import os
import torch
from detect import detect_image, detect_video
import cv2
from PIL import Image


app=Flask(__name__) # it's a common step to start with this
Swagger(app) # pass the App to Swagger

current_directory = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_directory, "model")

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model

@app.route('/') # must be written to define the root page or main page to display
# this will display a web page having welcome all in it
def welcome():
    return "Welcome All"

# a page for predicting one sample, can be used through Postman

@app.route('/detect',methods=["POST"]) # by default it's GET method because we will pass our features as parameters
def detect_A_sample():
    """
    Let's check face mask 
    ---
    parameters:
        
        - name: image
          in: formData
          type: file
          required: true
    produces:
        - image/*
    responses:
        200:
            description: ok
            content:
                image/jpg: {}

    """

    image = request.files.get("image")
    image = Image.open(image)
    image_path = os.path.join(current_directory, "image.jpg")
    image.save(image_path)
    detect_image(current_directory, model)
    image = open(image_path, 'rb')
    return  send_file(image, mimetype='image/jpg')

@app.route('/detect_video',methods=["POST"]) # by default it's GET method because we will pass our features as parameters
def detect_A_video():
    """
    Let's check face mask 
    ---
    parameters:
        
        - name: Video
          in: formData
          type: file
          required: true
    
    responses:
        200:
            description: ok
            

    """
    Video_path = os.path.join(current_directory, 'video.avi')
    if os.path.exists(Video_path):
        # Delete the file
        os.remove(Video_path)
    Video = request.files.get('Video')
    Video = Video.read()
    with open(Video_path, 'wb') as f:
        f.write(Video)
    detect_video(current_directory, model)   
    return "Go check the created video"
    

if __name__=='__main__':
    app.run()