import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (pretrained)
model = YOLO("yolov8n.pt")

# Function to determine if eye is open or closed
def eye_state(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Thresholding
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    # Count white pixels (iris/closed area)
    white_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size

    ratio = white_pixels / total_pixels

    # Heuristic condition
    if ratio > 0.25:
        return "CLOSED", ratio
    else:
        return "OPEN", ratio


def process_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found")
        return

    results = model(img)

    closed_count = 0
    open_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # YOLO COCO class 0 = person (no eye class)
            # So we simulate by cropping upper face region
            if cls == 0:  
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                face = img[y1:y2, x1:x2]

                h, w, _ = face.shape

                # Approximate eye region (upper part of face)
                eye_region = face[0:int(h/2), :]

                state, ratio = eye_state(eye_region)

                if state == "CLOSED":
                    closed_count += 1
                    color = (0, 0, 255)
                else:
                    open_count += 1
                    color = (0, 255, 0)

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, state, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    total = open_count + closed_count

    if total == 0:
        print("No face detected")
        return

    perclos = closed_count / total
    open_percentage = (open_count / total) * 100
    closed_percentage = (closed_count / total) * 100

    print("\n===== RESULT =====")
    print(f"Open Percentage   : {open_percentage:.2f}%")
    print(f"Closed Percentage : {closed_percentage:.2f}%")
    print(f"PERCLOS           : {perclos:.2f}")

    # Show output
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---- MAIN ----
if __name__ == "__main__":
    image_path = r"C:\Users\AKALYA\OneDrive\Pictures\Saved Pictures\images (2).jpeg"
    process_image(image_path)