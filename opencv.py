import numpy as np 
import cv2
import os 
import pickle
import random




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)
print(os.getcwd())

#import Face_train.py


image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
#profile_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_profileface.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels={}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap= cv2.VideoCapture(0)


while (True):
#    Face_train()
    ret,frame = cap.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5 , minNeighbors=5)
    #side = profile_cascade.detectMultiScale(
    #    gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces :

        #print('value for faces :')
        #print(x, y, w, h)
        roi_gray= gray [y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0,255,0)
            border = 2
            cv2.putText(frame,name,(x,y),font,1,color,border,cv2.LINE_AA)
            repup_dir = os.path.join(image_dir,name)
            img_color=str(random.random())+'_color.png'
            img_gray=str(random.random())+'_gray_.png'
            cv2.imwrite(os.path.join(repup_dir,img_gray),roi_gray)
            cv2.imwrite(os.path.join(repup_dir,img_color),roi_color)
        
        color = (255,0,0)#BGR
        border = 2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame, (x,y),(end_cord_x,end_cord_y),color,border)

    cv2.imshow ('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
