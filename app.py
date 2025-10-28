import streamlit as st
from utils.pose_extractor import extract_pose
from utils.feature_engineering import extract_features
from utils.model_loader import load_model

model = load_model()
uploaded_file = st.file_uploader("動画をアップロード", type=["mp4"])
if uploaded_file:
    video_path = save_temp_video(uploaded_file)
    pose_array = extract_pose(video_path)
    features = extract_features(pose_array)
    prediction = model.predict(features)

    st.write(f"フォーム評価（最終フレーム）: {prediction[-1]}")
