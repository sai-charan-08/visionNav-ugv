from ultralytics import YOLO
import cv2
import os

# ============================================================
# VISIONNAV - UGV PERCEPTION AND NAVIGATION
# ============================================================

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

MODEL_PATH = "models/yolo11n-seg.pt"
IMAGE_PATH = "input/test.jpeg"

# Minimum confidence required for an object to be considered
CONFIDENCE_THRESHOLD = 0.60


# ------------------------------------------------------------
# 2. LOAD YOLO SEGMENTATION MODEL
# ------------------------------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ------------------------------------------------------------
# 3. LOAD IMAGE
# ------------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load input/test.jpeg")
    print("Make sure test.jpeg is inside the input folder.")
    exit()

print("Image loaded successfully.")


# Get image dimensions
height, width = image.shape[:2]

# Divide image into 3 navigation zones
left_boundary = width // 3
right_boundary = (width * 2) // 3


# ------------------------------------------------------------
# 4. RUN YOLO INFERENCE
# ------------------------------------------------------------

print("\nRunning YOLO inference...")

results = model(
    image,
    conf=0.50,
    verbose=False
)

result = results[0]


# ------------------------------------------------------------
# 5. PROCESS DETECTIONS
# ------------------------------------------------------------

detected_objects = []

left_blocked = False
center_blocked = False
right_blocked = False


for box in result.boxes:

    # Get confidence
    confidence = float(box.conf[0])

    # --------------------------------------------------------
    # IMPORTANT:
    # Ignore weak detections
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    # Get class ID
    class_id = int(box.cls[0])

    # Get class name
    class_name = model.names[class_id]

    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Calculate center of bounding box
    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2

    # --------------------------------------------------------
    # Determine which section the object occupies
    # --------------------------------------------------------

    if object_center_x < left_boundary:
        position = "LEFT"
        left_blocked = True

    elif object_center_x < right_boundary:
        position = "CENTER"
        center_blocked = True

    else:
        position = "RIGHT"
        right_blocked = True

    # Store object information
    detected_objects.append({
        "name": class_name,
        "confidence": confidence,
        "position": position,
        "box": (x1, y1, x2, y2)
    })


# ------------------------------------------------------------
# 6. PRINT PERCEPTION RESULTS
# ------------------------------------------------------------

print("\n======================================")
print("       VISIONNAV UGV PERCEPTION")
print("======================================")

print(f"Objects detected: {len(detected_objects)}")

if len(detected_objects) == 0:

    print("No reliable objects detected.")

else:

    for obj in detected_objects:

        print(
            f"- {obj['name']} "
            f"(confidence: {obj['confidence']:.2f}) "
            f"[{obj['position']}]"
        )


# ------------------------------------------------------------
# 7. NAVIGATION DECISION
# ------------------------------------------------------------

if not detected_objects:

    navigation_decision = "MOVE FORWARD"

elif center_blocked:

    # Something is directly in front of the UGV

    if not left_blocked:
        navigation_decision = "TURN LEFT"

    elif not right_blocked:
        navigation_decision = "TURN RIGHT"

    else:
        navigation_decision = "STOP - PATH BLOCKED"

elif left_blocked and not right_blocked:

    navigation_decision = "TURN RIGHT"

elif right_blocked and not left_blocked:

    navigation_decision = "TURN LEFT"

else:

    navigation_decision = "MOVE FORWARD"


# ------------------------------------------------------------
# 8. PRINT NAVIGATION DECISION
# ------------------------------------------------------------

print("\n--------------------------------------")
print("NAVIGATION STATUS")
print("--------------------------------------")

print(f"Left blocked   : {left_blocked}")
print(f"Center blocked : {center_blocked}")
print(f"Right blocked  : {right_blocked}")

print("\nNAVIGATION DECISION:", navigation_decision)

print("======================================\n")


# ------------------------------------------------------------
# 9. DRAW NAVIGATION ZONES
# ------------------------------------------------------------

# Draw vertical boundaries
cv2.line(
    image,
    (left_boundary, 0),
    (left_boundary, height),
    (255, 255, 255),
    2
)

cv2.line(
    image,
    (right_boundary, 0),
    (right_boundary, height),
    (255, 255, 255),
    2
)


# Zone labels
cv2.putText(
    image,
    "LEFT",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.putText(
    image,
    "CENTER",
    (left_boundary + 20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

cv2.putText(
    image,
    "RIGHT",
    (right_boundary + 20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)


# ------------------------------------------------------------
# 10. DRAW RELIABLE DETECTIONS
# ------------------------------------------------------------

for obj in detected_objects:

    x1, y1, x2, y2 = obj["box"]

    class_name = obj["name"]
    confidence = obj["confidence"]
    position = obj["position"]

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    # Create label
    label = f"{class_name} {confidence:.2f}"

    # Draw label
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# ------------------------------------------------------------
# 11. DISPLAY NAVIGATION DECISION
# ------------------------------------------------------------

cv2.putText(
    image,
    "VISIONNAV - UGV",
    (20, height - 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    image,
    f"Objects: {len(detected_objects)}",
    (20, height - 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    image,
    f"Navigation: {navigation_decision}",
    (20, height - 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)


# ------------------------------------------------------------
# 12. SAVE RESULT
# ------------------------------------------------------------

output_folder = "runs"

os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "visionnav_result.jpg"
)

cv2.imwrite(output_path, image)

print(f"Result saved to: {output_path}")


# ------------------------------------------------------------
# 13. DISPLAY IMAGE
# ------------------------------------------------------------

cv2.imshow(
    "VisionNav - UGV Navigation",
    image
)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()