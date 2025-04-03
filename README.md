# Semantic-Segmentation

## Configuration for traing Custom dataset

1. Check the constant/config/__init__ to conifgure.
2. Make sure the dataset contains all the name as mentioned in config.
3. Change SELECTED_CLASS, CLASS_RGB_VALUES based on dataset you have.
4. Add models in models directry and import model Class in model_training and model_prediction

## How to run
1. Create python environment
            
            python3 -m venv myenv

2. Install cuda version pytorch

             pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

3. Install all requirements using

            pip install -r requirements.txt

    or

            python install .

4. Run the GUI using Streamlit

            streamlit run app.py