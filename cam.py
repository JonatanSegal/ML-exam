import os
import cv2
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as image

from tensorflow import keras




model = keras.models.load_model('Card_scanner_75.h5')
vid = cv2.VideoCapture(0)
targetSize = 64
color = 'rgb'

while (True):
    # Capture the video frame
    # by frame

    ret, frame = vid.read()


    # If needed, convert the frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(frame, 'The card is', (5,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,(64,62,59), 4, 2)

    # Display the resulting frame
    cv2.imshow('Camera feed', frame)

    predictImg = plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    test_image = image.load_img(predictImg, target_size=[targetSize,targetSize], color_mode=color)

    test_image = image.img_to_array(test_image) # convert image to array
    test_image = np.expand_dims(test_image,axis=0) # add one extra dimension to hold batch.
    # axis=0 means that a new dimension will be added, such that test_image.shape goes from (28, 28, 1) to (1, 28, 28, 1). This is required by Tensorflow.

    result = model.predict(test_image/255.0) # remember to divide each pixel value by 255.0

    print("result is: " + str(result[0][0]))


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()