import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch  # Import torch for device handling
import warnings

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Load YOLO model
model = YOLO('yolov8n.pt')  # Ensure model file is in the correct path
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Streamlit app
st.title("YOLO Object Detection App")

# File uploader for image input
img = st.file_uploader("Upload an Image for Detection", type=["jpg", "jpeg", "png"])

if img:
    # Convert the uploaded image to a format suitable for YOLO
    image = Image.open(img)
    image = np.array(image)

    # Perform inference
    results = model(image)  # YOLO inference
    result = results[0]  # Get the first (and likely only) result

    # Annotate and display the image
    annotated_img = result.plot()  # Create annotated image
    st.image(annotated_img, caption="Detected Objects", use_container_width=True)

    # Print detection summary
    st.write("Detection Summary:")
    for box in result.boxes:
        # Extract class and confidence as Python data types
        cls = int(box.cls.cpu().numpy())  # Convert to integer
        conf = float(box.conf.cpu().numpy())  # Convert to float
        st.write(f"Class: {cls}, Confidence: {conf:.2f}")





