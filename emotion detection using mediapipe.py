import cv2
import mediapipe as mp
from fer import FER
import csv
import os
from datetime import datetime

# Initialize FER detector with better face detection
fer_detector = FER(mtcnn=True)

# MediaPipe setups
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# CSV logging
log_filename = "C:/projects/emtion detector/emotion_log.csv"
if not os.path.exists(log_filename):
    with open(log_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Top Emotion", "Score", "All Emotions"])
 
# Start webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    #frame = cv2.flip(frame, 1)  # Mirror the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    mesh_results = face_mesh.process(frame_rgb)

    h, w, _ = frame.shape

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + width), min(h, y + height)
            face_crop = frame[y1:y2, x1:x2]

            try:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                emotions = fer_detector.detect_emotions(face_rgb)

                if emotions:
                    emotion_data = emotions[0]["emotions"]
                    # Filter to reduce false "neutral" dominance
                    emotion_data = {k: v for k, v in emotion_data.items() if v >= 0.05}
                    if emotion_data:
                        top_emotion = max(emotion_data, key=emotion_data.get)
                        score = emotion_data[top_emotion]
                    else:
                        top_emotion, score = "neutral", 0.0
                        emotion_data = {"neutral": 1.0}
                else:
                    top_emotion, score = "neutral", 0.0
                    emotion_data = {"neutral": 1.0}

                text = f"{top_emotion} ({score:.2f})"

                # Log to CSV
                with open(log_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([timestamp, top_emotion, f"{score:.2f}", emotion_data])

            except Exception as e:
                top_emotion = "neutral"
                score = 0.0
                emotion_data = {"neutral": 1.0}
                text = f"Error: {e}"

            # Draw bounding box and emotion text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Draw cross at face center
            cx = x1 + (x2 - x1) // 2
            cy = y1 + (y2 - y1) // 2
            cv2.line(frame, (cx, y1), (cx, y2), (255, 255, 0), 1)
            cv2.line(frame, (x1, cy), (x2, cy), (255, 255, 0), 1)

            # Draw emotion bars
            bar_y = y1
            for emotion, value in sorted(emotion_data.items(), key=lambda x: -x[1])[:5]:
                bar_length = int(value * 100)
                cv2.rectangle(frame, (x2 + 10, bar_y), (x2 + 10 + bar_length, bar_y + 20), (100, 100, 255), -1)
                cv2.putText(frame, f"{emotion} {value:.2f}", (x2 + 10, bar_y + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                bar_y += 25

    # Draw face landmarks (dots/mesh)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    # Show the frame
    cv2.imshow("Emotion Detection + Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
