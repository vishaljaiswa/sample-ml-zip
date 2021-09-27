from flask import Flask,render_template,Response
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("face_mask.model")


camera=cv2.VideoCapture(0)


def generate_frames():
    while True:
            
        success,frame=camera.read()
        if not success:
            break
        else:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (Type1, Type2,Type3,Type4) = pred
				
                if Type1 > Type2 and Type1>Type3 and Type1>Type4:
                    label="Mask Wear Ok"
                elif Type2>Type3 and Type2>Type4:
                    label="Mask ON Chin"
                elif Type3>Type4:
                    label="Mask not Wear Properly"
                else:
                    label="Please Weear Mask"


                color = (0, 255, 0) if label == "Type1" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(Type1, Type2,Type3,Type4) * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

application = Flask(__name__)
#application = app

@application.route('/')
def index():
    return render_template('video.html')

@application.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    application.run(debug=True, use_reloader=False,host="172.31.95.61",port=8000)