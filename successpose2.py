import cv2 as cv 
import numpy as np
import winsound
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

inWidth = 368
inHeight = 368
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

jumping_pairs = [["RHip", "RKnee"], ["RKnee", "RAnkle"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"]]

jump_threshold = 0.8
jump_velocity_threshold = 800
jump_acceleration_threshold = 2000

def pose():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FPS, 10)
    cap.set(3,800)
    cap.set(4,800)

    if not cap.isOpened():
        cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot Open Webcam")

    prev_points = None
    jump_count = 0

    def send_email(subject, body):
        try:
            email = 'retrohubmusic@gmail.com'
            password = 'howktwkhtnmmokgq'
            send_to_email = 'eagerxviper@gmail.com'

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

    while cv.waitKey(1)<0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]
        
        assert(len(BODY_PARTS) == out.shape[1])
        
        points = []

        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            
            points.append((int(x), int(y)) if conf > thr else None)
        
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)
            
            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]
        
            if points[idFrom] and points[idTo]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            
                if [partFrom, partTo] in jumping_pairs:
                    _, jump_height, _, _ = cv.minMaxLoc(out[0, idTo, :, :])
                    if jump_height > jump_threshold:
                        if prev_points is None:
                            prev_points = points
                            break # skip the check for jumping in the first iteration
                        else:
                            if prev_points[idTo] is not None and abs(prev_points[idTo][1] - points[idTo][1]) > jump_height / 2 and jump_height > 0.1:
                                jump_count += 1
                                prev_points = points
                                print("Total Movements: ", jump_count)
                                
                                if jump_count > 20:
                                    print("Suspicious Movements Detected")
                                    img_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                    img_path = os.path.join("images", img_name + ".jpg")
                                    cv.imwrite(img_path, frame)
                                    winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
                                    send_email("Suspicious Movements Detected", "Suspicious movements has been detected, please take action immediately.")
                                    jump_count = 0  # reset the jump count
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        cv.putText(frame, f"Date: {current_date}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"Time: {current_time}", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)   
            
        cv.imshow("Jump Detection", frame)


  