# Install required libraries (run once)


import cv2
import mediapipe as mp
import numpy as np


# Upload image
uploaded = files.upload()

# Get uploaded file name
image_path = list(uploaded.keys())[0]

# Read image
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Process image
results = face_mesh.process(rgb_image)

h, w, _ = image.shape

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:

        # Draw face outline
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS
        )

        # Eye landmark indices
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Draw left eye points
        for idx in LEFT_EYE:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # Draw right eye points
        for idx in RIGHT_EYE:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

# Show result
from matplotlib import pyplot as plt

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()