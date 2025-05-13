import cv2
import numpy as np
import math

# Define a region of interest (ROI) for hand detection
def get_roi(frame):
    return frame[100:400, 100:400]

# Preprocess the image for contour detection
def preprocess_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)      # Apply Gaussian blur
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Thresholding
    return thresh

# Count fingers using convex hull and defects
def count_fingers(thresh, roi):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0  # No hand detected

    # Find the largest contour (assumed to be the hand)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate convex hull and convexity defects
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0  # Not enough points for convexity defects

    defects = cv2.convexityDefects(largest_contour, hull)

    # Count fingers based on angles and distances
    if defects is None:
        return 0

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])

        # Calculate the angle between fingers
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)
        angle = math.acos((b*2 + c2 - a*2) / (2 * b * c)) * 57

        # Count as a finger if the angle is less than 90 degrees
        if angle < 90:
            finger_count += 1

    return finger_count

# Main function
def main():
    # Load the input video clip
    video_path = "input_video.mp4"  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Define region of interest (ROI) for hand detection
        roi = get_roi(frame)

        # Preprocess the ROI for contour detection
        thresh = preprocess_image(roi)

        # Count fingers
        finger_count = count_fingers(thresh, roi)

        # Display results
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)  # Draw ROI rectangle
        cv2.putText(frame, f"Fingers: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Threshold", thresh)
        cv2.imshow("Gesture Recognition", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
