import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

# 5th Tab: Building Detection from Predicted and Ground Truth Images
def process_building_image(pil_image, meter_per_pixel):
    """
    Processes an input PIL image to detect building contours.
    Returns:
        count: number of buildings detected,
        building_info: list of dictionaries with building number and area in sq ft,
        overlay_img_rgb: the image with drawn bounding boxes (in RGB for st.image).
    """
    # Convert image to grayscale
    cv_img = np.array(pil_image.convert('L'))
    # Threshold to obtain binary image (adjust threshold value as needed)
    ret, thresh = cv2.threshold(cv_img, 127, 255, cv2.THRESH_BINARY)
    # Find external contours (assumes each contour corresponds to a building)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    building_info = []
    # Convert original image to BGR for drawing colored bounding boxes
    overlay_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    for idx, cnt in enumerate(contours):
        # Filter out very small contours as noise (you may adjust the threshold)
        area_pixels = cv2.contourArea(cnt)
        if area_pixels < 50:
            continue
        # Compute bounding box for the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Convert pixel area to square meters, then to square feet.
        area_sq_m = area_pixels * (meter_per_pixel ** 2)
        area_sq_ft = area_sq_m * 10.7639
        building_info.append({
            "Building Number": len(building_info) + 1,
            "Area (sq ft)": f"{area_sq_ft:.4f}"
        })
    
    count = len(building_info)
    # Convert overlay image from BGR to RGB for display in Streamlit
    overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    
    return count, building_info, overlay_img_rgb

def building_detection_tab():
    st.header("Building Detection from Prediction & Ground Truth Images")
    
    st.markdown("### Conversion Factor")
    st.write("Please provide the conversion factor (meters per pixel) for your image. This value is used to convert the pixel area of a building into square feet.")
    meter_per_pixel = st.number_input("Meters per pixel", min_value=0.0001, value=0.3, step=0.01)
    
    st.markdown("---")
    st.markdown("### Upload Images")
    st.write("Upload the predicted image and the ground truth image (for aerial building extraction).")
    
    pred_file = st.file_uploader("Upload Predicted Image", type=["png", "jpg", "jpeg", "tiff", "tif"], key="pred_img")
    gt_file = st.file_uploader("Upload Ground Truth Image", type=["png", "jpg", "jpeg", "tiff", "tif"], key="gt_img")
    
    if pred_file is not None and gt_file is not None:
        # Open images using PIL
        pred_image = Image.open(pred_file)
        gt_image = Image.open(gt_file)
        
        st.markdown("### Preview Uploaded Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(pred_image, caption="Predicted Image", use_container_width=True)
        with col2:
            st.image(gt_image, caption="Ground Truth Image", use_container_width=True)
        
        # Process each image for building detection
        st.markdown("---")
        st.markdown("### Processed Bounding Box Images & Counts")
        pred_count, pred_buildings, pred_overlay = process_building_image(pred_image, meter_per_pixel)
        gt_count, gt_buildings, gt_overlay = process_building_image(gt_image, meter_per_pixel)
        
        st.subheader("Predicted Image - Building Detection")
        st.image(pred_overlay, caption="Predicted Image with Bounding Boxes", use_container_width=True)
        st.write(f"**Number of Buildings Detected (Predicted):** {pred_count}")
        
        st.subheader("Ground Truth Image - Building Detection")
        st.image(gt_overlay, caption="Ground Truth Image with Bounding Boxes", use_container_width=True)
        st.write(f"**Number of Buildings Detected (Ground Truth):** {gt_count}")
        
        # Create and display tables for each
        st.markdown("---")
        st.markdown("### Building Details (Predicted)")
        if pred_buildings:
            pred_df = pd.DataFrame(pred_buildings)
            st.dataframe(pred_df)
        else:
            st.write("No buildings detected in the predicted image.")
            
        st.markdown("### Building Details (Ground Truth)")
        if gt_buildings:
            gt_df = pd.DataFrame(gt_buildings)
            st.dataframe(gt_df)
        else:
            st.write("No buildings detected in the ground truth image.")

