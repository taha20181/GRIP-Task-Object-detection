import numpy as np
import argparse
import cv2 as cv
import time


print("[INFO] reading the input image and model...")

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True)
ap.add_argument("-p", "--prototxt", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.2)


args = vars(ap.parse_args())
time.sleep(3)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "cell phone", "laptop"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading Model...")

net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
time.sleep(3)

image = cv.imread(args['image'])
(h, w) = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image, (300,300)), 0.007843, (300,300), 127.5)


print("[INFO] computing object detections...")

net.setInput(blob)
detections = net.forward()
time.sleep(3)

for i in np.arange(0,detections.shape[2]):

    confidence = detections[0,0,i,2]

    if confidence > args["confidence"]:

        idx = int(detections[0,0,i,1])
        box = detections[0,0,i,3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[INFO] {}".format(label))

        cv.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)

        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv.putText(image, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)



cv.imshow("Detected", image)
cv.waitKey(5000)
cv.destroyAllWindows()