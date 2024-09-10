import cv2
import mediapipe as mp
import numpy as np
import time
from pygame import mixer


mixer.init()
mixer.music.load("C:\\Users\\Harshitha Devi S\\Downloads\\music.wav")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

def calculate_EAR(eye):
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    h = np.linalg.norm(eye[0] - eye[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_MAR(mouth):
    v1 = np.linalg.norm(mouth[1] - mouth[7])
    v2 = np.linalg.norm(mouth[2] - mouth[6])
    v3 = np.linalg.norm(mouth[3] - mouth[5])
    h = np.linalg.norm(mouth[0] - mouth[4])
    mar = (v1 + v2 + v3) / (2.0 * h)
    return mar

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

EYE_CLOSED_THRESHOLD = 0.25
DROWSY_TIME = 0.5
YAWN_THRESHOLD = 0.6
YAWN_FRAMES = 5

eye_closed_start = None
drowsy_alert = False
yawn_counter = 0
yawn_frames = 0
last_yawn_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mesh_points = np.array([
                [int(point.x * image.shape[1]), int(point.y * image.shape[0])]
                for point in face_landmarks.landmark
            ])

            left_eye = [mesh_points[i] for i in LEFT_EYE]
            right_eye = [mesh_points[i] for i in RIGHT_EYE]
            mouth = [mesh_points[i] for i in MOUTH]
            
            left_ear = calculate_EAR(np.array(left_eye))
            right_ear = calculate_EAR(np.array(right_eye))
            avg_ear = (left_ear + right_ear) / 2.0
            
            mar = calculate_MAR(np.array(mouth))

            cv2.polylines(image, [np.array(left_eye)], True, (0, 255, 0), 1)
            cv2.polylines(image, [np.array(right_eye)], True, (0, 255, 0), 1)
            cv2.polylines(image, [np.array(mouth)], True, (0, 255, 0), 1)

            if avg_ear < EYE_CLOSED_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start >= DROWSY_TIME:
                    drowsy_alert = True
            else:
                eye_closed_start = None
                drowsy_alert = False

            if mar > YAWN_THRESHOLD:
                yawn_frames += 1
                if yawn_frames >= YAWN_FRAMES:
                    current_time = time.time()
                    if current_time - last_yawn_time > 60:  # can Reset yawn counter every minute
                        yawn_counter = 0
                    yawn_counter += 1
                    last_yawn_time = current_time
                    yawn_frames = 0
            else:
                yawn_frames = 0

            if drowsy_alert:
                cv2.putText(image, "DROWSY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                mixer.music.play()

            cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"Yawns: {yawn_counter}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Drowsiness and Yawn Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()