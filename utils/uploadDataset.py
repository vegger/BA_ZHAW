import wandb
import pandas as pd
import argparse
import chardet
import os
from dotenv import load_dotenv

def upload_dataset_to_wandb(name: str, data_path: str, separator: str):
    load_dotenv()
    project_name = os.getenv('MAIN_PROJECT_NAME')
    run = wandb.init(project=project_name, job_type=f"dataset-upload-{name}")
    with open(data_path, 'rb') as file:
        result = chardet.detect(file.read())  # Read some bytes from the file
    encoding = result['encoding']
    df = pd.read_csv(data_path, sep=separator, encoding=encoding)
    df = df.astype(str)
    dataset = wandb.Artifact(name, type='dataset')
    dataset.add(wandb.Table(dataframe=df), name+"_table")
    run.log_artifact(dataset)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset to W&B.")
    parser.add_argument("--name", type=str, required=True, help="Name of the W&B artifact.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--separator", type=str, default="\t", help="Separator used in the dataset file (default: '\t').")
    
    args = parser.parse_args()
    
    upload_dataset_to_wandb(args.name, args.data_path, args.separator)