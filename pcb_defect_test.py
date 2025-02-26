import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('best_20250226.engine')

# Open the camera (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Get the class names from the model (they should be stored in the model's metadata)
class_names = model.names  # Access the class labels from the model

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Break the loop if no frame is returned
    if not ret:
        break

    # Run the model on the current frame
    #results = model.track(source=frame, verbose=False, device=0, stream=True, persist=True)
    results = model.track(source=frame, stream=True)
    for res in results:
        annotated_frame = res.plot()

    # Display the frame with the bounding boxes and labels drawn
    cv2.imshow('Frame', annotated_frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

