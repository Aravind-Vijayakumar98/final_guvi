# face_detection_app.py

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import tempfile
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ultralytics import YOLO  # Deep Learning enhancement

# Set Streamlit config
st.set_page_config(page_title="Human Face Detection App", layout="wide")
st.title("\U0001F9D1 Human Face Detection & Classification App")

# Sidebar Navigation
menu = st.sidebar.selectbox("Choose Function", ["Data", "EDA - Visual", "Prediction"])

# Load real face bounding box dataset
@st.cache_data
def load_face_data():
    return pd.read_csv("faces.csv")

df = load_face_data()

# Dummy classification column based on bounding box area
bbox_area = (df['x1'] - df['x0']) * (df['y1'] - df['y0'])
df['Face_Large'] = np.where(bbox_area > bbox_area.median(), 1, 0)
X = df[['width', 'height', 'x0', 'y0', 'x1', 'y1']]
y = df['Face_Large']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Load YOLOv8 general object detection model (fallback to "person")
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")  # Using general model as face-specific not found

yolo_model = load_yolo_model()

if menu == "Data":
    st.header("1. Dataset Viewer")
    st.subheader("Face Metadata (Real Dataset)")
    st.dataframe(df.head(20))

    st.subheader("Upload a single face image")
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(uploaded_image.getvalue())
            temp_path = temp_image.name

        image_name = uploaded_image.name
        st.success(f"Image '{image_name}' uploaded successfully.")

        image = cv2.imread(temp_path)
        if image is not None:
            results = yolo_model(temp_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            names = yolo_model.names
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    label = names[cls]
                    if label.lower() in ["face", "person"]:  # Fallback to person if face not detected
                        x0, y0, x1, y1 = map(int, box.xyxy[0])
                        cv2.rectangle(image_rgb, (x0, y0), (x1, y1), (0, 255, 0), 2)
            st.image(image_rgb, caption=f"Detected Faces in {image_name}", use_container_width=True)
        else:
            st.error("Failed to read image using OpenCV. It may be corrupted or not a supported format.")

    st.subheader("Model Performance Metrics (Face Size Classification)")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    st.json(metrics)

elif menu == "EDA - Visual":
    st.header("2. Exploratory Data Analysis")
    st.subheader("Face Bounding Box Size Distribution")
    df['area'] = (df['x1'] - df['x0']) * (df['y1'] - df['y0'])
    fig1 = px.histogram(df, x='area', nbins=30, title="Distribution of Face Bounding Box Area")
    st.plotly_chart(fig1)

    st.subheader("Image Resolution Distribution")
    fig2 = px.scatter(df, x='width', y='height', title="Image Resolution Scatter Plot")
    st.plotly_chart(fig2)

    st.subheader("Face Size by Image")
    fig3 = px.box(df, x='Face_Large', y='area', title="Box Plot of Bounding Box Area by Face Size Class")
    st.plotly_chart(fig3)

elif menu == "Prediction":
    st.header("3. Prediction Form")
    st.subheader("Enter Bounding Box & Image Dimensions")
    inputs = {}
    for col in ['width', 'height', 'x0', 'y0', 'x1', 'y1']:
        inputs[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    if st.button("Predict Face Size"):
        input_df = pd.DataFrame([inputs])
        result = clf.predict(input_df)[0]
        st.success("Prediction: " + ("Large Face" if result == 1 else "Small Face"))

st.sidebar.markdown("""
---
**Tags:**
- OpenCV
- YOLOv8
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Hyperparameter Tuning
- Deep Learning
""")
