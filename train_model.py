import cv2
import os
import numpy as np
from PIL import Image

# Setup recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    # This maps your photos to numbers (0, 1, 2)
    name_map = {"meghana": 0, "harish": 1, "sangeetha": 2}

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        
        # Get name from filename
        filename = os.path.split(imagePath)[-1].split(".")[0].lower()
        if filename in name_map:
            id = name_map[filename]
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    return faceSamples,ids

print("\n Training faces. Please wait...")
faces,ids = getImagesAndLabels('ImagesAttendance')
recognizer.train(faces, np.array(ids))

recognizer.write('trainer.yml') 
print(f"\n [INFO] {len(np.unique(ids))} faces trained successfully!")