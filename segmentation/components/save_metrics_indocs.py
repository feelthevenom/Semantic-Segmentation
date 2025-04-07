import streamlit as st
import os
import json
import pandas as pd
from docx import Document
from docx.shared import Inches

# Import necessary constants and configurations from your app modules
from segmentation.constant.config import TRAIN_METRIC_DIRR, TEST_METRIC_DIRR
from segmentation.entity.config_entity import ArtifactConfig

# Instantiate artifact config to get the model base path
artifact_config = ArtifactConfig()
model_base_path = artifact_config.trained_models_dirr

# Helper function to format float values to 4 decimal places
def format_val(val):
    if isinstance(val, (float, int)):
        return f"{val:.4f}"
    return str(val)

# Function to read a JSON file and return the data
def read_json_file(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error reading {filepath}: {e}")
        return None

# Function to process training metrics: identifies the best epoch (using validation accuracy)
# and sums the total training time.
def process_train_metrics(train_data):
    best_epoch = None
    best_val_acc = -1
    total_time = 0
    for record in train_data:
        epoch = record.get("epoch")
        valid_metrics = record.get("valid_metrics", {})
        valid_acc = valid_metrics.get("accuracy", 0)
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_epoch = epoch
        train_metrics = record.get("train_metrics", {})
        total_time += train_metrics.get("total_time", 0)
    return best_epoch, total_time

# Function to create a complete docx report for all datasets and models.
def create_doc_table_all(model_base_path):
    document = Document()
    document.add_heading("Model Metrics Report", level=1)

    # List dataset directories in model_base_path
    dataset_dirs = [d for d in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, d))]
    if not dataset_dirs:
        document.add_paragraph("No datasets found in the model base path.")
        output_path = "metrics_report_all.docx"
        document.save(output_path)
        return output_path

    for dataset in dataset_dirs:
        document.add_heading(f"Dataset: {dataset}", level=2)
        dataset_path = os.path.join(model_base_path, dataset)
        model_dirs = [m for m in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, m))]
        if not model_dirs:
            document.add_paragraph("No models found for this dataset.")
            continue

        for model in model_dirs:
            document.add_heading(f"Model: {model}", level=3)
            # Define paths for train and test metrics JSON files
            train_metrics_path = os.path.join(model_base_path, dataset, model, TRAIN_METRIC_DIRR)
            test_metrics_path = os.path.join(model_base_path, dataset, model, TEST_METRIC_DIRR)

            # Read JSON data for training and testing metrics
            train_data = read_json_file(train_metrics_path)
            test_data = read_json_file(test_metrics_path)

            # --- Training & Validation Metrics ---
            if train_data is not None:
                document.add_paragraph("Training and Validation Metrics:")
                table = document.add_table(rows=1, cols=8)
                hdr_cells = table.rows[0].cells
                hdr_cells[0].text = "Epoch"
                hdr_cells[1].text = "Train Dice Loss"
                hdr_cells[2].text = "Train IoU"
                hdr_cells[3].text = "Train Accuracy"
                hdr_cells[4].text = "Valid Dice Loss"
                hdr_cells[5].text = "Valid Accuracy"
                hdr_cells[6].text = "Valid Precision"
                hdr_cells[7].text = "Valid Recall"
                for record in train_data:
                    row_cells = table.add_row().cells
                    row_cells[0].text = format_val(record.get("epoch", "N/A"))
                    train_metrics = record.get("train_metrics", {})
                    valid_metrics = record.get("valid_metrics", {})
                    row_cells[1].text = format_val(train_metrics.get("dice_loss", "N/A"))
                    row_cells[2].text = format_val(train_metrics.get("iou_score", "N/A"))
                    row_cells[3].text = format_val(train_metrics.get("accuracy", "N/A"))
                    row_cells[4].text = format_val(valid_metrics.get("dice_loss", "N/A"))
                    row_cells[5].text = format_val(valid_metrics.get("accuracy", "N/A"))
                    row_cells[6].text = format_val(valid_metrics.get("precision", "N/A"))
                    row_cells[7].text = format_val(valid_metrics.get("recall", "N/A"))
                best_epoch, total_time = process_train_metrics(train_data)
                document.add_paragraph(f"Best Epoch (based on validation accuracy): {format_val(best_epoch)}")
                document.add_paragraph(f"Total Training Time: {format_val(total_time)} seconds")
            else:
                document.add_paragraph("No training metrics found.")

            # --- Testing Metrics ---
            if test_data is not None:
                document.add_paragraph("Test Metrics:")
                test_table = document.add_table(rows=1, cols=6)
                test_hdr_cells = test_table.rows[0].cells
                test_hdr_cells[0].text = "Dice Loss"
                test_hdr_cells[1].text = "IoU Score"
                test_hdr_cells[2].text = "Accuracy"
                test_hdr_cells[3].text = "Precision"
                test_hdr_cells[4].text = "Recall"
                test_hdr_cells[5].text = "Fscore"
                if isinstance(test_data, list) and len(test_data) > 0:
                    test_metrics = test_data[0].get("test_metrics", {})
                    row_cells = test_table.add_row().cells
                    row_cells[0].text = format_val(test_metrics.get("dice_loss", "N/A"))
                    row_cells[1].text = format_val(test_metrics.get("iou_score", "N/A"))
                    row_cells[2].text = format_val(test_metrics.get("accuracy", "N/A"))
                    row_cells[3].text = format_val(test_metrics.get("precision", "N/A"))
                    row_cells[4].text = format_val(test_metrics.get("recall", "N/A"))
                    row_cells[5].text = format_val(test_metrics.get("fscore", "N/A"))
                else:
                    document.add_paragraph("Test metrics data is empty.")
            else:
                document.add_paragraph("No test metrics found.")

            # Add a divider between models
            document.add_paragraph("-----------------------------")
        # Divider between datasets
        document.add_paragraph("====================================")
    
    output_path = "metrics_report_all.docx"
    document.save(output_path)
    return output_path

# New Tab for creating the doc table across datasets and models
def create_doc_table_tab():
    st.header("Create Doc Table")
    st.write("Click the Start button to generate the complete metrics report document from all dataset and model JSON metrics files.")

    if st.button("Start"):
        # Create the document by looping over all datasets and models
        output_docx = create_doc_table_all(model_base_path)
        st.success(f"Metrics report created: {output_docx}")

        # Provide a download button for the generated docx file
        with open(output_docx, "rb") as file:
            st.download_button("Download Report", data=file, file_name=output_docx, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")