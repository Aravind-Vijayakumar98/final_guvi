# Human Face Detection with YOLOv8

# Project Overview
This project detects human faces in images  using YOLOv8 and Python. The model is deployed using Streamlit for an interactive web app experience.

---

# Problem Statement
Detect and localize human faces in various images using object detection models.

---

# Objectives
- Load YOLOv8 model trained specifically for face detection
- Detect faces with bounding boxes
- Real-time detection via webcam
- Web interface using Streamlit

---

# Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Torch

---

# How to Run

### 1. Clone the repo & setup virtual environment


https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run face_detection_app.py


