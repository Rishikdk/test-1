from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('best.pt')

# Load video
video_path = '0'
cap = cv2.VideoCapture(video_path)

# Initialize tracker configuration
tracker_config = 'bytetrack.yaml'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection and tracking frame-by-frame
    results = model.track(source=frame, persist=True, conf=0.3, tracker=tracker_config)  # Adjust confidence threshold as needed

    # Loop through results and draw bounding boxes with unique IDs
    if results and len(results) > 0 and results[0].boxes is not None:  # Ensure results are not empty and boxes exist
        for box in results[0].boxes:  # Iterate over detected boxes
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates
            track_id = int(box.id.cpu().numpy()) if box.id is not None else -1  # Unique tracking ID
            class_id = int(box.cls.cpu().numpy())  # Class ID
            label = f"ID {track_id} - Class {class_id}"

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 ByteTrack Tracking", frame)

    # Introduce a delay to slow down the video
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Change 100 to control the delay (in milliseconds)
        break

cap.release()
cv2.destroyAllWindows()
