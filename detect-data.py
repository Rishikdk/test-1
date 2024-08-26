import cv2
import os
import numpy as np
from ultralytics import YOLO
import imagehash
from PIL import Image


# box overlap
def overlap(box1, box2):

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

# hash of an image.
def compute_hash(image):
     
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image)

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)  

# Create a folder to save images
output_folder = 'detected_faces'
os.makedirs(output_folder, exist_ok=True)

frame_count = 0  # To keep track of frames
tracked_objects = {}  # store object id
next_id = 1  

# Dictionary to store image hashes and corresponding IDs
image_hashes = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    current_objects = []  # List to store objects detected in the current frame

    
    for result in results:
        for box in result.boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Get class label, confidence, and bounding box
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            # Filter for face (class ID 0) or person (class ID 1)
            if class_id in [0, 1]:  # Adjust according to your model's class IDs
                current_objects.append((x1, y1, x2, y2, class_id, confidence))

    # Update tracked objects with current detections
    new_tracked_objects = {}
    for obj_id, (prev_box, obj_data) in tracked_objects.items():
        # Check if the previous object is in the current frame
        for box in current_objects:
            if overlap(prev_box, box[:4]):  # Check overlap with bounding box coordinates
                new_tracked_objects[obj_id] = (box[:4], {'class_id': box[4], 'confidence': box[5]})
                current_objects.remove(box)
                break

    # Assign new IDs to remaining objects
    for box in current_objects:
        x1, y1, x2, y2 = box[:4]
        cropped_img = frame[y1:y2, x1:x2]

        if cropped_img.size == 0:  # Ensure the cropped image is valid
            continue

        img_hash = compute_hash(cropped_img)

        # Check if this hash is already known
        matched_id = None
        for saved_hash, saved_id in image_hashes.items():
            if img_hash == saved_hash:
                matched_id = saved_id
                break

        if matched_id is not None:
            # Existing image, assign the same ID
            new_tracked_objects[matched_id] = (box[:4], {'class_id': box[4], 'confidence': box[5]})
        else:
            # New image, assign a new ID
            new_tracked_objects[next_id] = (box[:4], {'class_id': box[4], 'confidence': box[5]})
            image_hashes[img_hash] = next_id
            next_id += 1

    # Update tracked objects
    tracked_objects = new_tracked_objects

    # Draw bounding boxes and save images
    for obj_id, (box, obj_data) in tracked_objects.items():
        x1, y1, x2, y2 = box
        label = f"ID {obj_id}: {obj_data['confidence']:.2f}"
        
        # Crop the detected face/person
        cropped_img = frame[y1:y2, x1:x2]
        
        if cropped_img.size == 0:  # Ensure the cropped image is valid
            continue
        
        
        img_save_path = os.path.join(output_folder, f'{obj_id}.jpg')
        cv2.imwrite(img_save_path, cropped_img)

        # label on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # draw bounding boxes
    cv2.imshow('YOLOv8 Detection and Tracking', frame)

    frame_count += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
