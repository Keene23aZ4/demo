@st.cache_resource
def load_model():
    import joblib
    return joblib.load("ski_model.pkl")