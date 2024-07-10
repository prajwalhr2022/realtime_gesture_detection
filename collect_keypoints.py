import cv2
import numpy as np
import os
import mediapipe as mp
from extract_keypoints import extract_keypoints

# Define the data path and actions
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

# Create the directories if they don't exist
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize the holistic model from mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Start capturing video
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to capture image")
                        break

                    # Process the frame with mediapipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)

                    # Draw landmarks
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Draw face landmarks (if available)
                    if results.face_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                    # Draw pose landmarks
                    if results.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                    # Draw left hand landmarks
                    if results.left_hand_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Draw right hand landmarks
                    if results.right_hand_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Display information
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Exit on 'q' key press
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # Exit on 'q' key press
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Exit on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
