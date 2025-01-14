from flask import render_template, request
import os 
import cv2
from app.face_recognition import faceRecognitionPipeline


UPLOAD_FOLDER = 'static/upload'


def index():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        
        #save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path) # save image into upload folder
        
        # Get Prediction
        pred_image ,predictions = faceRecognitionPipeline(path)
        
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)
        
        return render_template('base.html', fileupload=True, path=path)
        
    return render_template('base.html', fileupload=False)