import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("ski_model.pkl")
