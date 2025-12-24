import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# Load FER-2013 CSV
df = pd.read_csv('fer2013.csv')

# Emotion labels (you can drop some later if you want)
label_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

X, y, usage_list = [], [], []

#iterate through all the samples in the FER-2013 dataframe
for _, row in tqdm(df.iterrows(), total=len(df)):
    #decoding image pixels by splitting string
    pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
    img = pixels.reshape(48, 48)

    #grayscale to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    #running mediapipe face mesh
    result = mp_face.process(img_rgb)
    if not result.multi_face_landmarks:
        continue

    #extracting landmarks
    landmarks = result.multi_face_landmarks[0].landmark

    #landmarks to np array
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    # normalize based off of nose tip coordinates (1 is nosetip)
    coords -= coords[1]

    # scale by face size (33 (left eye corner), 263 (right eye corner))
    scale = np.linalg.norm(coords[33] - coords[263])
    coords /= scale

    #saving the flattened coordinates and the emotion label
    X.append(coords.flatten())
    y.append(row["emotion"])
    usage_list.append(row["Usage"])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save("X_landmarks.npy", X)
np.save("y_labels.npy", y)
np.save("usage.npy", np.array(usage_list))

print("saved:", X.shape, y.shape)
