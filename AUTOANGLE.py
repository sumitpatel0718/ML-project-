import cv2
import numpy as np
import mediapipe as mp
import math
import os
from matplotlib import pyplot as plt

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

# Function to process a single image and calculate angles
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}, skipping.")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect keypoints
    results = pose.process(image_rgb)

    # Extracting key body landmarks if detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract required points
        shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
        elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.shape[0])
        wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * image.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * image.shape[0])
        
        hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1],
               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])
        knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * image.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * image.shape[0])
        ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.shape[1],
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.shape[0])

        # Calculate angles
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)
        back_angle = calculate_angle(shoulder, hip, knee)

        # Draw pose landmarks on image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Display image with landmarks
        plt.figure(figsize=(8, 6))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title(f"Pose Detection: {os.path.basename(image_path)}")
        plt.show()

        # Return the calculated angles
        return {
            "Image": os.path.basename(image_path),
            "Elbow Angle": round(elbow_angle, 2),
            "Knee Angle": round(knee_angle, 2),
            "Back Angle": round(back_angle, 2),
        }
    else:
        print(f"Pose not detected in {image_path}, skipping.")
        return None

# Folder containing images
folder_path = r"C:\Users\tanis\Downloads\PLAYER"  # Replace with your folder path

# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Process all images in the folder
valid_results = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Process only image files
        image_path = os.path.join(folder_path, filename)
        result = process_image(image_path)
        if result:
            valid_results.append(result)

# Print results for all valid images
print("\nFinal Angles for Processed Images:")
for res in valid_results:
    print(res)
