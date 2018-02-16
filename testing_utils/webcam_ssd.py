import keras
import pickle
#from videotest import VideoTest
import cv2
import numpy as np
import time
import sys
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import pickle
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer
import sys
sys.path.append("..")
from ssd_utils import BBoxUtility

sys.path.append("..")
from ssd import SSD300 as SSD

input_shape = (300, 300, 3)
conf_thresh = 0.6
# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../weights_SSD300.hdf5')

cap = cv2.VideoCapture(2)
bbox_util = BBoxUtility(NUM_CLASSES)

class_colors = []
for i in range(0, NUM_CLASSES):
    # This can probably be written in a more elegant manner
    hue = 255 * i / NUM_CLASSES
    col = np.zeros((1, 1, 3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128  # Saturation
    col[0][0][2] = 255  # Value
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)

ret, img = cap.read()
# Compute aspect ratio of image
imgh, imgw, channels = img.shape
imgar = imgw / imgh
im_size = (input_shape[0], input_shape[1])
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    st = time.time()

    resized = cv2.resize(img, im_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    to_draw = cv2.resize(resized, (int(input_shape[0] * imgar)*3, input_shape[1]*3))

    # Use model to predict
    inputs = [image.img_to_array(rgb)]
    tmp_inp = np.array(inputs)
    x = preprocess_input(tmp_inp)
    y = model.predict(x)

    results = bbox_util.detection_out(y)

    if len(results) > 0 and len(results[0]) > 0:
        # Interpret output, only one frame is used
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * to_draw.shape[1]))
            ymin = int(round(top_ymin[i] * to_draw.shape[0]))
            xmax = int(round(top_xmax[i] * to_draw.shape[1]))
            ymax = int(round(top_ymax[i] * to_draw.shape[0]))

            # Draw the box on top of the to_draw image
            class_num = int(top_label_indices[i])
            cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax),
                          class_colors[class_num], 2)
            text = class_names[class_num] + " " + ('%.2f' % top_conf[i])

            text_top = (xmin, ymin - 10)
            text_bot = (xmin + 80, ymin + 5)
            text_pos = (xmin + 5, ymin)
            cv2.rectangle(to_draw, text_top, text_bot, class_colors[class_num], -1)
            cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Display the resulting frame
    print('Elapsed time = {}'.format(time.time() - st))
    cv2.imshow("detection", to_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
