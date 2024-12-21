import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' does not exist.")
    exit()

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    print(f"Processing directory: {dir_}")

    for img_path in os.listdir(dir_path):
        img_file = os.path.join(dir_path, img_path)

        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            print(f"Skipping non-image file: {img_file}")
            continue

        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Failed to load image {img_file}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
        else:
            print(f"Warning: No hand landmarks detected in {img_file}")

output_file = 'data.pickle'
try:
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Data successfully saved to {output_file}")
except Exception as e:
    print(f"Error saving data: {e}")
