import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="スキー動作分類", layout="wide")
st.title("🎿 スキー動作分類モデルの推論アプリ")

# ファイルアップロード
st.sidebar.header("📂 ファイルアップロード")
model_file = st.sidebar.file_uploader("モデルファイル (.pkl)", type=["pkl"])
features_file = st.sidebar.file_uploader("特徴量ファイル (.npy)", type=["npy"])
encoder_file = st.sidebar.file_uploader("LabelEncoderファイル (.pkl)", type=["pkl"], help="任意")

# 推論処理
if model_file and features_file:
    model = joblib.load(model_file)
    features = np.load(features_file)

    # 推論
    predictions = model.predict(features)

    # ラベル復元（任意）
    if encoder_file:
        le = joblib.load(encoder_file)
        labels = le.inverse_transform(predictions)
    else:
        labels = predictions

    st.success("✅ 推論が完了しました！")

    # 表示：フレームごとのラベル
    st.subheader("🧪 フレームごとの推論ラベル")
    for i, label in enumerate(labels):
        st.write(f"フレーム {i}: {label}")

    # グラフ表示
    st.subheader("📈 ラベルの時系列可視化")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(labels, marker="o", linestyle="-", markersize=2)
    ax.set_title("フレームごとの推論ラベル")
    ax.set_xlabel("フレーム")
    ax.set_ylabel("ラベル")
    st.pyplot(fig)

else:
    st.info("左のサイドバーからモデルと特徴量ファイルをアップロードしてください。")
