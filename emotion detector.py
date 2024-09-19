import cv2
from fer import FER

# Initialize the FER detector
emotion_detector = FER()

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to grab frame")
        break
    
    # Analyze the frame for emotions
    emotions = emotion_detector.detect_emotions(frame)
    
    # Display emotions on the frame
    for emotion in emotions:
        (x, y, w, h) = emotion['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        emotion_text = emotion['emotions']
        max_emotion = max(emotion_text, key=emotion_text.get)
        cv2.putText(frame, f"{max_emotion}: {emotion_text[max_emotion]:.2f}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
