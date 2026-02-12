import cv2
import mediapipe as mp
import time

# Import your feature extraction function
from feature_extraction import extract_pose_features

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def run_pose_estimation():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return

    prev_time = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        print("✅ Pose estimation started. Press 'q' to quit.")

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

            # Draw skeleton if detected
            if results.pose_landmarks:
                # 🔹 Extract features here
                features = extract_pose_features(results.pose_landmarks)
                print("Features:", features)

                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                status_text = "Pose: Detected"
            else:
                status_text = "Pose: Not Detected"

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # Show status text and FPS
            cv2.putText(image_bgr, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image_bgr, f"FPS: {int(fps)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Pose Estimation - Press Q to Quit", image_bgr)

            # Break on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting pose estimation...")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_estimation()
