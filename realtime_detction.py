import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.optimizers import Adam
import mediapipe as mp
from mediapipe_holistic import mediapipe_detection, draw_styled_landmarks
from extract_keypoints import extract_keypoints

actions = np.array(['hello', 'thanks', 'iloveyou'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Custom optimizer handling
def custom_adam():
    return Adam(learning_rate=0.001)  # Adjust parameters as needed

# Register custom objects including initializers and optimizers
with tf.keras.utils.custom_object_scope({
    'Orthogonal': Orthogonal,
    'OrthogonalInitializer': Orthogonal,
    'Adam': custom_adam
}):
    model = load_model('C:/Users/Admin/PycharmProjects/pythonProject/final project/action.h5', compile=False)
    # Compile the model with a new optimizer
    model.compile(optimizer=custom_adam(), loss='categorical_crossentropy', metrics=['accuracy'])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

sequence = []
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        
        # Check and draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            image = prob_viz(res, actions, image, colors)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
