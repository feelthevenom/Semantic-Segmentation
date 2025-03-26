from segmentation.components.model_training import ModelTraining
from segmentation.components.model_prediction import ModelPrediction

# Example usage
if __name__ == "__main__":

    # trainer = ModelTraining()
    # trainer.train_all_datasets()

    predict_pipe = ModelPrediction()
    predict_pipe.predict()