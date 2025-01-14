import numpy as np
import sklearn
import pickle
import cv2



# Load all models
haar = cv2.CascadeClassifier(r'./model/haarcascade_frontalface_default.xml') # cascade classifier

with open('./model/model_svm.pickle', 'rb') as file: 
      
    model_svm=pickle.load(file) # machine learning model (SVM)
    

with open('./model/pca_dict.pickle', 'rb') as file: 
      
    pca_models=pickle.load(file) # pca dictionary


model_pca = pca_models['pca'] #pca model 
model_face_arr = pca_models['mean_face'] #Mean face

def faceRecognitionPipeline(filename):

    # Step-01: read Image
    img = cv2.imread(filename) #BGR

    # Step-02: convert into gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step-03: crop the face (using haar cascase classifier)
    faces = haar.detectMultiScale(gray, 1.5, 3)
    predictions = []
    for x, y, w, h in faces:
        #cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)
        roi = gray[y:y+h, x:x+w]
        
        # Step-04: normalization (0-1)
        roi = roi/255.0
        # Step-05: resize images (100,100) 
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi, (100,100))
        else:
            roi_resize = cv2.resize(roi, (100,100))           
        
        # Step-06: Flattening (1x10000) 
        roi_reshape = roi_resize.reshape(1,10000)
        
        # Step-07: subtract with mean  
        roi_mean = roi_reshape - model_face_arr #subtrace face with mean face    
        
        # Step-08: get eigen image (apply roi_mean to pca)
        eigen_iamge = model_pca.transform(roi_mean)
        
        # Step-09 Eigen Image for Visualization 
        eig_img = model_pca.inverse_transform(eigen_iamge)
        
        # Step-10: pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_iamge)  
        prob_score = model_svm.predict_proba(eigen_iamge)
        prob_score_max = prob_score.max()
        
        # Step-11: generate report
        text = "%s"%(results[0])
        
        # defining color based on results
        if results[0] == 'male':
            color = (255, 255, 0)
        else:
            color = (255, 0, 255)
        cv2.rectangle(img, (x,y),(x+w, y+h), color,3)
        cv2.rectangle(img, (x,y-40), (x+w, y), color, -1)
        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 3 ,(255,255,255),6)
        
        output = {
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name':results[0],
            'score':prob_score_max
        }
        predictions.append(output)
    return img, predictions
            