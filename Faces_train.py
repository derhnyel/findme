import os
from PIL import Image
import numpy as np
import cv2
import pickle


recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(
    'cascade/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

current_id =0

label_ids ={}

image_dir = os.path.join(BASE_DIR,"images")

y_labels=[]

x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","_")#.lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
                id_ = label_ids[label]
                #print(label_ids)    
            #print(label,path)
            #y_label.append()
            #x_train.append()
            pil_image = Image.open(path).convert('L')
            image_array = np.array(pil_image,'uint8')
            #print(image_array)
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)
                  

with open('labels.pickle','wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))

recognizer.save('trainer.yml')
