import cv2

# Initialize video capture object
cap = cv2.VideoCapture("cars.mp4")

# Create background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Apply Gaussian blur to the frame
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # Apply background subtraction to the blurred frame
    fgmask = fgbg.apply(blurred)

    # Apply thresholding to the foreground mask
    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
