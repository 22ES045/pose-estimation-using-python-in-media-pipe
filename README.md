# pose-estimation-using-python-in-media-pipe
pose estimation project built with python and media pipe for real time human key points detection.
Pose Estimation using MediaPipe
 is a computer vision project that detects and tracks human body landmarks in real-time using a webcam or video input.

MediaPipe provides highly optimized and accurate ML solutions for pose estimation with 33 key body landmarks
**Feautures**
- Real-time human pose detection
- Works with webcam or video files
- Built using Python + MediaPipe + OpenCV
- Lightweight and easy to run
**Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/python-mediapipe-pose-estimation.git
   cd python-mediapipe-pose-estimation
 **  Install dependencies
 Install media pipe opencv-Python
**Code for Posture Recongnization** 
import cv2
import mediapipe as mp

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

# Function to check finger open/close (Right Hand)
def finger_status(hand_landmarks):
    status = {}

    # Thumb: For right hand, thumb open if TIP.x < MCP.x
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
        status["thumb"] = "open"
    else:
        status["thumb"] = "closed"

    # Other fingers: open if TIP.y < PIP.y
    status["index"] = "open" if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y else "closed"
    status["middle"] = "open" if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y else "closed"
    status["ring"] = "open" if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y else "closed"
    status["pinky"] = "open" if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y else "closed"

    return status

# Function to recognize gestures
def get_gesture(hand_landmarks):
    fingers = finger_status(hand_landmarks)

    # Open Palm 🖐️
    if all(val == "open" for val in fingers.values()):
        return "Open Palm"

    # Fist (WOW) ✊
    if all(val == "closed" for val in fingers.values()):
        return "WOW"

    # Thumbs Up 👍
    if fingers["thumb"] == "open" and all(fingers[f] == "closed" for f in ["index", "middle", "ring", "pinky"]):
        return "Thumbs Up - Done"

    # Victory ✌️
    if fingers["index"] == "open" and fingers["middle"] == "open" and all(fingers[f] == "closed" for f in ["ring", "pinky", "thumb"]):
        return "Victory"

    # OK 👌
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    if dist < 0.04 and fingers["middle"] == "closed" and fingers["ring"] == "closed" and fingers["pinky"] == "closed":
        return "OK"

    return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)

            # Bounding box
            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.putText(image, gesture, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Right Hand Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
**Author**
Ram Singh 
https://github.com/22ES045/pose-estimation-using-python-in-media-pipe/edit/main/README.md
