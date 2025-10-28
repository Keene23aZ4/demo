import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_features(pose_array):
    features = []
    for frame in pose_array:
        try:
            left_knee = calculate_angle(frame[23], frame[25], frame[27])
            right_knee = calculate_angle(frame[24], frame[26], frame[28])
            torso_tilt = calculate_angle(frame[11], frame[23], frame[24])
            features.append([left_knee, right_knee, torso_tilt])
        except:
            features.append([np.nan, np.nan, np.nan])
    return np.array(features)
