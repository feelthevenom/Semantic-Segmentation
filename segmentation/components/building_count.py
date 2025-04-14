import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def detect_buildings(mask_pil: Image.Image, meter_per_pixel: float):
    """
    Detects building contours in a binary mask image.
    Returns:
        building_info: list of dicts with Building Number and Area (sq ft)
        bboxes: list of tuples (x, y, w, h) for each building
        mask_thresh: binary mask as numpy array
    """
    # Convert mask to grayscale array
    mask_gray = np.array(mask_pil.convert('L'))
    # Threshold to binary
    _, mask_thresh = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    # Find external contours
    contours, _ = cv2.findContours(mask_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    building_info = []
    bboxes = []

    for cnt in contours:
        area_pixels = cv2.contourArea(cnt)
        # Filter out small contours as noise
        if area_pixels < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        building_number = len(building_info) + 1
        # Calculate area in square meters and convert to square feet
        area_sq_m = area_pixels * (meter_per_pixel ** 2)
        area_sq_ft = area_sq_m * 10.7639
        building_info.append({
            "Building Number": building_number,
            "Area (sq ft)": f"{area_sq_ft:.4f}"
        })
        bboxes.append((x, y, w, h))

    return building_info, bboxes, mask_thresh


def building_detection_tab():
    """
    Streamlit tab for building detection:
    - Upload predicted mask and original image
    - Overlay mask on original and draw bounding boxes
    - Show table of building areas
    - Allow user to select a building to view its clipped region
    """
    st.header("Building Detection and Visualization")

    # Conversion factor input
    st.markdown("### Conversion Factor")
    st.write(
        "Provide the conversion factor (meters per pixel) to convert pixel area to square feet."
    )
    meter_per_pixel = st.number_input(
        "Meters per pixel",
        min_value=0.0001,
        value=0.3,
        step=0.01
    )

    st.markdown("---")
    st.markdown("### Upload Images")
    st.write(
        "Upload the predicted mask image and the original aerial image (RGB)."
    )

    pred_file = st.file_uploader(
        "Upload Predicted Mask Image",
        type=["png", "jpg", "jpeg", "tiff"],
        key="pred_mask"
    )
    orig_file = st.file_uploader(
        "Upload Original Image",
        type=["png", "jpg", "jpeg", "tiff"],
        key="orig_img"
    )

    if pred_file and orig_file:
        # Load images
        pred_mask = Image.open(pred_file)
        orig_image = Image.open(orig_file).convert('RGB')

        # Preview images
        st.markdown("### Preview Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(orig_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(pred_mask, caption="Predicted Mask Image", use_container_width=True)

        st.markdown("---")
        # Detect buildings in mask
        building_info, bboxes, mask_thresh = detect_buildings(pred_mask, meter_per_pixel)

        # Prepare overlay: convert original to BGR for OpenCV
        orig_bgr = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
        # Create color mask (red) where mask is present
        mask_color = np.zeros_like(orig_bgr)
        mask_color[mask_thresh == 255] = [0, 0, 255]
        # Blend original and mask
        overlay = cv2.addWeighted(orig_bgr, 0.7, mask_color, 0.3, 0)

        # Draw bounding boxes and labels
        for info, (x, y, w, h) in zip(building_info, bboxes):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                str(info["Building Number"]),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Convert back to RGB for Streamlit
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Display overlay
        st.subheader("Original Image with Mask Overlay and Bounding Boxes")
        st.image(overlay_rgb, use_container_width=True)
        st.write(f"**Number of Buildings Detected:** {len(building_info)}")

        st.markdown("---")
        # Show details and allow selection
        if building_info:
            df = pd.DataFrame(building_info)
            st.dataframe(df)
            selected = st.selectbox(
                "Select Building",
                [b["Building Number"] for b in building_info],
                key="building_select"
            )
            sel_idx = selected - 1
            sel_info = building_info[sel_idx]
            sel_bbox = bboxes[sel_idx]
            x, y, w, h = sel_bbox

            # Show selected building info
            st.write(
                f"📌 **Building {selected}** - Area: {sel_info['Area (sq ft)']} sq ft"
            )

            # Clip the region from the original image
            orig_rgb = np.array(orig_image)
            clip = orig_rgb[y:y+h, x:x+w]
            st.image(
                clip,
                caption=f"Clipped Original Image - Building {selected}",
                use_container_width=True
            )
        else:
            st.write("No buildings detected in the predicted mask.")
