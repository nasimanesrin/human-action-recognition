import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculate angle between three points: a, b, c
    Angle at point b
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def extract_pose_features(landmarks):
    """
    landmarks: MediaPipe pose landmarks
    returns: feature vector (list of floats)
    """

    # Convert landmarks to list of (x, y)
    points = [(lm.x, lm.y) for lm in landmarks.landmark]

    # Select important joints by index (MediaPipe Pose)
    # 11: left shoulder, 13: left elbow, 15: left wrist
    # 12: right shoulder, 14: right elbow, 16: right wrist
    # 23: left hip, 25: left knee, 27: left ankle
    # 24: right hip, 26: right knee, 28: right ankle

    features = []

    # Angles: left arm
    features.append(calculate_angle(points[11], points[13], points[15]))  # left elbow
    features.append(calculate_angle(points[12], points[14], points[16]))  # right elbow

    # Angles: legs
    features.append(calculate_angle(points[23], points[25], points[27]))  # left knee
    features.append(calculate_angle(points[24], points[26], points[28]))  # right knee

    # Distances (normalized)
    def dist(p1, p2):
        return math.dist(p1, p2)

    features.append(dist(points[11], points[23]))  # left shoulder to left hip
    features.append(dist(points[12], points[24]))  # right shoulder to right hip
    features.append(dist(points[15], points[11]))  # left wrist to left shoulder
    features.append(dist(points[16], points[12]))  # right wrist to right shoulder

    return features
