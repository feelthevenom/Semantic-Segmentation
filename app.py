import streamlit as st
import numpy as np
import os
import pandas as pd
import json

from segmentation.components.model_training import ModelTraining
from segmentation.components.model_prediction import ModelPrediction
from segmentation.entity.config_entity import ArtifactConfig
from segmentation.constant.config import TRAIN_METRIC_DIRR, TEST_METRIC_DIRR


artifact_config = ArtifactConfig()
model_path = artifact_config.trained_models_dirr

print(model_path)
os.makedirs(model_path, exist_ok= True)

# Fetch available model folders dynamically
models = [model for model in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, model))]



# --- Streamlit App ---
if __name__ == "__main__":

    st.title("Semantic Segmentation")
    tab1, tab2, tab3 = st.tabs(["Training", "Testing", "Prediction"])

    # ====================== TRAINING ==============================
    with tab1:
        st.header("Train all the models")

        trigger_train = st.button("Train the models")
        trainer = ModelTraining()
        if trigger_train:
            trainer.train_all_datasets()

        if trainer.train_all_datasets:
            st.info("Training is Completed! Refresh page to see the metrics")
        # ---- Dynamic Model Tabs ----
        if models:
            model_tabs = st.tabs([f"{model} Metrics" for model in models])

            for idx, model_name in enumerate(models):
                with model_tabs[idx]:
                    st.subheader(f"{model_name} - Training Metrics")

                    metrics_path = os.path.join(model_path, model_name, TRAIN_METRIC_DIRR)

                    refresh = st.button("🔄 Refresh Metrics", key=f"refresh_{idx}")

                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as f:
                            metrics_data = json.load(f)

                        # Convert to DataFrame
                        records = []
                        for entry in metrics_data:
                            epoch = entry["epoch"]
                            train = entry["train_metrics"]
                            valid = entry["valid_metrics"]
                            records.append({
                                "epoch": epoch,
                                "train_iou": train["iou_score"],
                                "valid_iou": valid["iou_score"],
                                "train_dice": train["dice_loss"],
                                "valid_dice": valid["dice_loss"],
                                "train_acc": train["accuracy"],
                                "valid_acc": valid["accuracy"]
                            })

                        df = pd.DataFrame(records)

                        # Plots
                        st.line_chart(df.set_index("epoch")[["train_iou", "valid_iou"]])
                        st.line_chart(df.set_index("epoch")[["train_dice", "valid_dice"]])
                        st.line_chart(df.set_index("epoch")[["train_acc", "valid_acc"]])

                        # Table
                        st.subheader("Metrics Data")
                        st.dataframe(df)

                        if refresh:
                            st.rerun()
                    else:
                        st.warning(f"No metrics file found for {model_name}. Please train first.")

    # ====================== TESTING ==============================
    with tab2:
        st.header("Test All the dataset")
        trigger_test = st.button("Test the models")
        if trigger_test:
            tester = ModelPrediction()
            tester.predict()

        if models:
            model_tabs = st.tabs([f"{model} Metrics" for model in models])

            for idx, model_name in enumerate(models):
                with model_tabs[idx]:
                    st.subheader(f"{model_name} - Testing Metrics")

                    metrics_path = os.path.join(model_path, model_name, TEST_METRIC_DIRR)
                    st.info(metrics_path)

                    if os.path.exists(metrics_path):
                        with open(metrics_path, "r") as f:
                            metrics_data = json.load(f)

                        # Convert to DataFrame
                        records = []
                        for entry in metrics_data:
                            test = entry["test_metrics"]
                            records.append({
                                "test_iou": test["iou_score"],
                                "test_dice": test["dice_loss"],
                                "test_acc": test["accuracy"]
                            })

                        df = pd.DataFrame(records)
                        # Table
                        st.subheader("Metrics Data")
                        st.dataframe(df)

                    else:
                        st.warning(f"No metrics file found for {model_name}. Please train first.")
    # ====================== PREDICTION ==============================
    with tab3:
        st.header("Image Prediction 🚀")

        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tiff"])

        model_choice = st.selectbox("Select Model", options=["MSU_Net", "BMSU_Net"])

        if uploaded_file is not None:
            # Save uploaded image temporarily
            img_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(img_path, use_container_width =True)

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    from segmentation.components.prediction import Prediction
                    predictor = Prediction(model_name=model_choice)
                    _, mask = predictor.predict_image(img_path)

                    with col2:
                        st.subheader("Predicted Mask")
                        st.image(mask, use_container_width =True)

