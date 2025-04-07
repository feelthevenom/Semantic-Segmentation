import streamlit as st
import numpy as np
import os
import pandas as pd
import json
from docx import Document

from segmentation.components.model_training import ModelTraining
from segmentation.components.model_prediction import ModelPrediction
from segmentation.components.save_metrics_indocs import create_doc_table_tab
from segmentation.components.building_count import building_detection_tab
from segmentation.entity.config_entity import ArtifactConfig, DatasetConfig
from segmentation.constant.config import TRAIN_METRIC_DIRR, TEST_METRIC_DIRR, BATCH_NUM, EPOCHS, LR_RATE

artifact_config = ArtifactConfig()
dataset_config = DatasetConfig()
# dataset_path now contains all dataset directories
dataset_dirs = dataset_config.all_dataset
model_base_path = artifact_config.trained_models_dirr
os.makedirs(model_base_path, exist_ok=True)

# --- Initialize Session State for Hyperparameters if not already set ---
if "model_params" not in st.session_state:
    st.session_state["model_params"] = {}

# --- Streamlit App ---
if __name__ == "__main__":

    st.title("Semantic Segmentation Dashboard")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Training", "Testing", "Prediction", "create doc table", "Building Detection"
    ])

    # ====================== TRAINING ==============================
    with tab1:
        st.header("Train all the models")

        # --- Sidebar for Training Parameters ---
        st.sidebar.header("Training Parameters")
        # Use the dataset directories from dataset_config.all_dataset directly
        dataset_dirs = dataset_config.all_dataset  
        if dataset_dirs:
            selected_dataset = st.sidebar.selectbox("Select Dataset for Params", options=dataset_dirs, key="selected_dataset")
            
            # Fetch model names directly from ModelTraining class (self.models)
            trainer = ModelTraining()
            model_dirs = list(trainer.models.keys())
            if model_dirs:
                selected_model = st.sidebar.selectbox("Select Model for Params", options=model_dirs, key="selected_model")
            else:
                st.sidebar.info("No models found in the training configuration.")
                selected_model = None
        else:
            st.sidebar.info("No datasets available.")
            selected_dataset = None
            selected_model = None


        # Only show hyperparameter inputs if both dataset and model are selected
        if selected_dataset and selected_model:
            # Create a unique key for the current dataset-model combination
            param_key = f"{selected_dataset}_{selected_model}"
            # If parameters for this key do not exist, initialize with default values
            if param_key not in st.session_state.model_params:
                st.session_state.model_params[param_key] = {
                    "batch_num": BATCH_NUM,
                    "epochs": EPOCHS,
                    "lr_rate": 0.01  # updated default LR_RATE set to 0.01
                }

            # Callback to update session state when any parameter changes
            def update_model_params():
                st.session_state.model_params[param_key] = {
                    "batch_num": st.session_state[f"{param_key}_batch"],
                    "epochs": st.session_state[f"{param_key}_epochs"],
                    "lr_rate": st.session_state[f"{param_key}_lr_rate"]
                }

            user_batch_num = st.sidebar.number_input(
                "Batch Number", min_value=1,
                value=st.session_state.model_params[param_key]["batch_num"],
                key=f"{param_key}_batch",
                on_change=update_model_params
            )
            user_epochs = st.sidebar.number_input(
                "Epochs", min_value=1,
                value=st.session_state.model_params[param_key]["epochs"],
                key=f"{param_key}_epochs",
                on_change=update_model_params
            )
            user_lr_rate = st.sidebar.number_input(
                "Learning Rate", min_value=0.000001,
                value=st.session_state.model_params[param_key]["lr_rate"],
                step=0.01, format="%.6f",
                key=f"{param_key}_lr_rate",
                on_change=update_model_params
            )
        else:
            user_batch_num = BATCH_NUM
            user_epochs = EPOCHS
            user_lr_rate = 0.01

        trigger_train = st.button("Train the models")
        trainer = ModelTraining()
        if trigger_train:
            # Pass the user-defined hyperparameters to the training routine.
            trainer.train_all_datasets(lr_rate=user_lr_rate, epochs=user_epochs, batch_num=user_batch_num)
            st.success("✅ Training Completed. Refresh to view metrics!")

        # ---- Dynamic Dataset and Model Tabs for Training Metrics ----
        if dataset_dirs:
            dataset_tabs = st.tabs([f"Dataset: {d}" for d in dataset_dirs])
            for d_idx, dataset_name in enumerate(dataset_dirs):
                with dataset_tabs[d_idx]:
                    st.subheader(f"Metrics for Dataset: {dataset_name}")
                    # Build the full path to the dataset artifact folder
                    dataset_artifact_path = os.path.join(model_base_path, dataset_name)
                    if os.path.exists(dataset_artifact_path):
                        model_dirs = [m for m in os.listdir(dataset_artifact_path)
                                    if os.path.isdir(os.path.join(dataset_artifact_path, m))]
                        if model_dirs:
                            model_tabs = st.tabs([f"{m}" for m in model_dirs])
                            for m_idx, model_name in enumerate(model_dirs):
                                with model_tabs[m_idx]:
                                    st.subheader(f"{model_name} - Training Metrics")
                                    metrics_path = os.path.join(model_base_path, dataset_name, model_name, TRAIN_METRIC_DIRR)
                                    st.info(f"📄 Metrics Path: {metrics_path}")
                                    refresh = st.button("🔄 Refresh", key=f"refresh_{d_idx}_{m_idx}")
                                    
                                    if os.path.exists(metrics_path):
                                        with open(metrics_path, "r") as f:
                                            metrics_data = json.load(f)
                                    else:
                                        metrics_data = []  # No metrics yet
                                    
                                    if metrics_data:
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
                                        st.line_chart(df.set_index("epoch")[["train_iou", "valid_iou"]])
                                        st.line_chart(df.set_index("epoch")[["train_dice", "valid_dice"]])
                                        st.line_chart(df.set_index("epoch")[["train_acc", "valid_acc"]])
                                        st.dataframe(df)
                                        
                                        if refresh:
                                            st.rerun()
                                    else:
                                        st.warning("Model is not trained yet. Please train or refresh after training.")
                        else:
                            st.info("No models found in this dataset. Please train first.")
                    else:
                        st.info(f"No trained models available for '{dataset_name}'. Please train the model to generate metrics.")


    # ====================== TESTING ==============================
    with tab2:
        st.header("Test all the models")
        trigger_test = st.button("Run Testing on all datasets")
        if trigger_test:
            tester = ModelPrediction()
            tester.predict()
            st.success("✅ Testing Completed. Refresh to view metrics!")
        dataset_dirs = [d for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]
        if dataset_dirs:
            dataset_tabs = st.tabs([f"Dataset: {d}" for d in dataset_dirs])
            for d_idx, dataset_name in enumerate(dataset_dirs):
                with dataset_tabs[d_idx]:
                    st.subheader(f"Testing Metrics for Dataset: {dataset_name}")
                    model_dirs = [m for m in os.listdir(os.path.join(model_base_path, dataset_name)) 
                                  if os.path.isdir(os.path.join(model_base_path, dataset_name, m))]
                    if model_dirs:
                        model_tabs = st.tabs([f"{m}" for m in model_dirs])
                        for m_idx, model_name in enumerate(model_dirs):
                            with model_tabs[m_idx]:
                                st.subheader(f"{model_name} - Testing Metrics")
                                metrics_path = os.path.join(model_base_path, dataset_name, model_name, TEST_METRIC_DIRR)
                                st.info(f"📄 Metrics Path: {metrics_path}")
                                if os.path.exists(metrics_path):
                                    with open(metrics_path, "r") as f:
                                        metrics_data = json.load(f)
                                    records = []
                                    for entry in metrics_data:
                                        test = entry["test_metrics"]
                                        records.append({
                                            "test_iou": test["iou_score"],
                                            "test_dice": test["dice_loss"],
                                            "test_acc": test["accuracy"]
                                        })
                                    df = pd.DataFrame(records)
                                    st.dataframe(df)
                                else:
                                    st.warning(f"No testing metrics found for {model_name} on {dataset_name}. Please test first.")
                    else:
                        st.info("No models found in this dataset.")

    # ====================== PREDICTION ==============================
    with tab3:
        st.header("Predict on an Image 🚀")
        # Dynamically get datasets and models for prediction
        dataset_dirs = [d for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]
        selected_dataset = st.selectbox("Select Dataset", options=dataset_dirs, key="pred_selected_dataset")
        model_dirs = []
        if selected_dataset:
            model_dirs = [m for m in os.listdir(os.path.join(model_base_path, selected_dataset))
                          if os.path.isdir(os.path.join(model_base_path, selected_dataset, m))]
        selected_model = st.selectbox("Select Model", options=model_dirs, key="pred_selected_model")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "tiff"])
        if uploaded_file is not None:
            # Save uploaded image temporarily
            img_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Image")
                st.image(img_path, use_container_width=True)
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    from segmentation.components.prediction import Prediction
                    predictor = Prediction(dataset_name=selected_dataset, model_name=selected_model)
                    _, mask = predictor.predict_image(img_path)
                    with col2:
                        st.subheader("Predicted Mask")
                        st.image(mask, use_container_width=True)

        with tab4:
            create_doc_table_tab()

        with tab5:
            building_detection_tab()