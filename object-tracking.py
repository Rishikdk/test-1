import os
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Load video
video_path = 'D:/Project-6th/test-1/test-vedio/IMG_1237.MOV'
cap = cv2.VideoCapture(video_path)

# Directory to save images
output_dir = 'D:/Project-6th/output_images/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

ret = True
frame_count = 0  # To keep track of frame numbers

# Read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # Detect and track objects
        results = model.track(frame, persist=True)

        # Plot results and save images for each tracked object
        for result in results:
            # Extract tracking information
            if hasattr(result, 'tracking_id') and result.tracking_id is not None:
                tracking_id = result.tracking_id

                for bbox, class_id in zip(result.boxes.xyxy, result.boxes.cls):
                    # Convert bounding box to integers
                    bbox = list(map(int, bbox))
                    x1, y1, x2, y2 = bbox

                    # Create a directory for each tracking ID if it doesn't exist
                    tracking_dir = os.path.join(output_dir, f'ID_{int(tracking_id)}')
                    if not os.path.exists(tracking_dir):
                        os.makedirs(tracking_dir)

                    # Crop the image for the detected object
                    cropped_img = frame[y1:y2, x1:x2]

                    # Save the cropped image with a unique name
                    img_name = f'{tracking_dir}/frame_{frame_count}.jpg'
                    cv2.imwrite(img_name, cropped_img)

                    # Draw bounding box and label on the original frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{model.names[int(class_id)]} {int(tracking_id)}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Increment the frame count
        frame_count += 1

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
