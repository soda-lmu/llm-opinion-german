import argparse
import pandas as pd
from src.paths import TEXT_GEN_DIR
from src.bert.bert_classifier import BertClassifier
import os

def classify_experiments(experiment_data_path,model_name, threshold=0.5, device='cpu'):

    model = BertClassifier(model_name=model_name, device=device)

    print(f"Running on data in {experiment_data_path}")
    model.classify_experiment_folder(folder_path=experiment_data_path, threshold=threshold)
    print(f"Predictions saved to {experiment_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify experiments using BERT model")
    parser.add_argument('--experiment_data_path', type=str, required=True, help='Folder path for the experiment data')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu" or "cuda")')
    parser.add_argument('--model_name', type=str, default='bert_mixed_coarse_resample20240708_195103', help='The name of the model')

    args = parser.parse_args()

    classify_experiments(experiment_data_path=args.experiment_data_path, model_name=args.model_name,threshold=args.threshold, device=args.device)