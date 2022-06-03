import cv2
import numpy as np

from tensorflow.keras.models import load_model#to load our trained model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

model=load_model('model_building_defects_vgg16.h5')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        (W, H) = (None, None)

        # while True:
        grabbed, fr = self.video.read()

        # if not grabbed:
        #     break
        
        if W is None or H is None:
            (H, W) = fr.shape[:2]
        
        output=fr.copy()
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (224, 224))
        

        x=image.img_to_array(fr)
        x=np.expand_dims(fr, axis=0)
        img_data=preprocess_input(x)
        result = np.argmax(model.predict(img_data), axis=-1)
        index=['crack','flakes','roof']
        result=str(index[result[0]])
        
        cv2.putText(output, "Defect: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                    2, (0,255,255), 1)
        cv2.imshow("Text", output)
        print(result)
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()