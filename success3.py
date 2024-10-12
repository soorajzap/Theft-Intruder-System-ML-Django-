import cv2
import matplotlib.pyplot as plt
import winsound
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels=[]
file_name='labels.txt'
with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)



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
        send_to_email = 'soorajzap@gmail.com'

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
cv2.destroyAllWindows()




