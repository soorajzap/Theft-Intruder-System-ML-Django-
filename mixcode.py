from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt
import winsound
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

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

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
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

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

model = cv2.dnn_DetectionModel(frozen_model,config_file)
maskNet = load_model("mask_detector.model")

classLabels=[]
file_name='labels.txt'
with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


def record():

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    font_scale=3
    font=cv2.FONT_HERSHEY_PLAIN

    def send_email(subject, body):
        try:
            email = 'retrohubmusic@gmail.com'
            password = 'howktwkhtnmmokgq'
            send_to_email = 'akhilbiju2001@gmail.com'

            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = send_to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email, password)
            text = msg.as_string()
            server.sendmail(email, send_to_email, text)
            server.quit()
            print("Email sent successfully")
        except Exception as e:
            print(f"Error sending email: {e}")


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'recordings/{datetime.now().strftime("%d-%H-%M")}.avi', fourcc, 20.0, (640,480))

    while True:
        ret,frame = cap.read()
        ClassIndex, confidece, bbox, = model.detect(frame, confThreshold=0.55)

        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 0, 255) if label == "Mask" else (0, 255, 0)

            if label == "Mask":
                print("Mask Detected")
                winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
                send_email("Mask detected", "A person wearing mask has been detected, please take action immediately.")
                    
        if (len(ClassIndex)!=0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                if(ClassInd<=80):
                    if classLabels[ClassInd-1] == "knife" or classLabels[ClassInd-1] == "gun":
                        # Trigger alarm, e.g. play a sound and send email
                        print("ALARM: Weapon detected!")
                        winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
                        send_email("Weapon detected", "A weapon has been detected, please take action immediately.")
                        cv2.rectangle(frame, boxes, (0, 0, 255), 2) # Draw red border
                    else:
                        cv2.rectangle(frame, boxes, (255, 0, 0), 2) # Draw blue border
                        cv2.putText(frame, classLabels[ClassInd-1],(boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv2.putText(frame, f"Date: {current_date}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {current_time}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)

        out.write(frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

        # Save frame as a video
        

    cap.release()
    out.release()
    vs.stop()
cv2.destroyAllWindows()










