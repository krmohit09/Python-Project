
import cv2
import time
import numpy as np
import winsound  # For beep sound on Windows
from scipy.spatial import distance as dist
import mediapipe as mp

 #Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 60
counter = 0
alarm_on = False

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Eye landmark indices (Mediapipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Start webcam
print("video start")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            def get_eye_points(indices):
                return [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]

            leftEye = get_eye_points(LEFT_EYE_IDX)
            rightEye = get_eye_points(RIGHT_EYE_IDX)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Decide color based on eye state
            if ear < EAR_THRESHOLD:
                eye_color = (0, 0,255)  # Red for closed/blinking eyes
                counter += 1

                if counter >= CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        winsound.Beep(1000, 1000)

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                eye_color = (0, 255, 0)  # Green for open eyes
                counter = 0
                alarm_on = False

            # Draw eyes with respective color
            cv2.polylines(frame, [np.array(leftEye)], True, eye_color, 2)
            cv2.polylines(frame, [np.array(rightEye)], True, eye_color, 2)

            # Show EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
