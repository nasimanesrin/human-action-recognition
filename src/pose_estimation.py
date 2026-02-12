import cv2
import mediapipe as mp
import time

from feature_extraction import extract_pose_features
from classifier import ActionClassifier

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def run_pose_estimation():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return

    # Load trained classifier
    print("📦 Loading action classifier...")
    classifier = ActionClassifier("models/action_classifier.pkl")

    prev_time = 0
    current_action = "None"
    current_confidence = 0.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        print("✅ Real-time action recognition started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame. Exiting...")
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process pose
            results = pose.process(image_rgb)

            # Convert back to BGR for display
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Default status
            status_text = "Pose: Not Detected"

            # If pose detected, extract features and predict action
            if results.pose_landmarks:
                features = extract_pose_features(results.pose_landmarks)

                # Predict action
                label, confidence = classifier.predict(features)
                current_action = str(label)
                current_confidence = float(confidence)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                status_text = "Pose: Detected"

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # Display texts
            cv2.putText(image_bgr, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(image_bgr, f"Action: {current_action}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.putText(image_bgr, f"Confidence: {current_confidence:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(image_bgr, f"FPS: {int(fps)}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Real-Time Action Recognition - Press Q to Quit", image_bgr)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation()
