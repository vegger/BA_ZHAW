import wandb
import os
from dotenv import load_dotenv


load_dotenv()
PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
# alternative name: Vanilla_Allele, Physico_Gene, Physico_Allele
artifact = wandb.Artifact(name="Vanilla_Gene", type="dataset")

artifact.add_dir('/teamspace/studios/this_studio/BA/paired/negative_samples/gene/', name='Gene')
run = wandb.init(project=PROJECT_NAME, job_type="Upload Dataset", entity="ba-zhaw")
run.log_artifact(artifact)
