import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import LaneDetector

WEIGHT_PATH = "weight/curvelanes_res34.pth"

st.set_page_config(page_title="Lane Detection", layout="wide")
st.title("Lane Detection")
st.write("Group Term Project — Ultra-Fast Lane Detection v2 (CurveLanes ResNet-34)")


@st.cache_resource
def load_detector(weight_path, device):
    return LaneDetector(weight_path, device=device)


st.sidebar.header("Settings")
draw_style = st.sidebar.radio("Draw style", ["all", "ego"], index=0,
                              help="'ego' keeps only the two lanes bordering the driving lane.")
lane_color = st.sidebar.selectbox("Lane color", ["blue", "green", "red", "yellow", "white"], index=0)
line_width = st.sidebar.slider("Line width", 1, 10, 4)
device = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
st.sidebar.info("Model: UFLDv2 ResNet-34 trained on CurveLanes.")

detector = load_detector(WEIGHT_PATH, device)

uploaded_file = st.file_uploader("Browse or drop an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.spinner("Running lane detection..."):
        annotated = detector.annotate(image.copy(), draw_style=draw_style,
                                      lane_color=lane_color, width=line_width)
        coords = detector.predict(image)

    with col2:
        st.subheader("Lane Detection Output")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.success("Analysis complete.")
    st.write("### Technical Summary")
    st.write(f"- **Algorithm:** Ultra-Fast Lane Detection v2 — hybrid anchor ordinal classification")
    st.write(f"- **Backbone:** ResNet-34 (CurveLanes pretrained, F1 81.34)")
    st.write(f"- **Input resolution:** {image.shape[1]}×{image.shape[0]}")
    st.write(f"- **Lanes detected:** {len(coords)} (draw style: `{draw_style}`)")
else:
    st.info("Please upload an image to start detection.")
