import cv2

# Open the video file
cap = cv2.VideoCapture('path_to_your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each frame
    # e.g., apply object detection and tracking here

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
