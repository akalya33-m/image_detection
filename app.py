

import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Load image from system
image_path = "face.jpg"   # 👉 change this to your image path
frame = cv2.imread(image_path)

# Check if image loaded
if frame is None:
    print("Error: Image not found")
    exit()

# Convert to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Process image
results = face_mesh.process(rgb)

# Draw landmarks
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = frame.shape

        # Example feature points
        points = [33, 133, 362, 263, 1, 13]  # eyes, nose, mouth

        for idx in points:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

# Show output
cv2.imshow("Face Detection (Image)", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()