import cv2
from keras.models import load_model
from PIL import Image
import os
import numpy as np
from keras.models import load_model

model=load_model('Braintumour2904.h5')
image=cv2.imread('C:\\Project clg\\pred\\pred0.jpg')
img=Image.fromarray(image)
img=img.resize((64,64))

img = np.array(img)

input_img=np.expand_dims(img, axis=0)

print(input_img)

result=model.predict(input_img)
classes_x=np.argmax(result,axis=1)
print(classes_x)

if classes_x == [1]:
    print("There is brain tumor")
else:
    print("There is no tumor")
