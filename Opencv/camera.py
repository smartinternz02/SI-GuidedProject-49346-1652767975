import cv2 # opencv library
from tensorflow.keras.models import load_model#to load our trained model
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

model=load_model('model_building_defects_vgg16.h5')
print("Loaded model from disk")
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
#writer = None
(W, H) = (None, None)

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    x=image.img_to_array(frame)
    x=np.expand_dims(frame, axis=0)
    img_data=preprocess_input(x)
    result = np.argmax(model.predict(img_data), axis=-1)
    index=['crack','flakes','roof']
    result=str(index[result[0]])
    cv2.putText(output, "Defect: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                2, (0,255,255), 1)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
# release the file pointers
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()