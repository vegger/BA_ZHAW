from vanilla_model import VanillaModel
from dataclass_paired_vanilla import PairedVanilla
import torch
from dotenv import load_dotenv
import os
import wandb
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
import pytorch_lightning as pl

EMBEDDING_SIZE = 1024
BATCH_SIZE = 128
NUM_WORKERS = 0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_NAME = "VanillaModel"

def column_to_dictionray(df, column_name):
    list_of_column = df[column_name].unique()
    dictionary = {}
    for index, item in enumerate(list_of_column):
        dictionary[item] = index
    return dictionary


def get_embed_len(df, column_name):
    list_of_column = df[column_name].unique()
    return len(list_of_column)


class PadCollate:
    def __init__(self, seq_max_length):
        self.seq_max_length = seq_max_length

    def pad_collate(self, batch):
        epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings = [], [], []
        v_alpha, j_alpha, v_beta, j_beta = [], [], [], []
        mhc = []
        task = []
        labels = []

        for item in batch:
            epitope_embeddings.append(item["epitope_embedding"])
            tra_cdr3_embeddings.append(item["tra_cdr3_embedding"])
            trb_cdr3_embeddings.append(item["trb_cdr3_embedding"])
            v_alpha.append(item["v_alpha"])
            j_alpha.append(item["j_alpha"])
            v_beta.append(item["v_beta"])
            j_beta.append(item["j_beta"])
            mhc.append(item["mhc"])
            task.append(item["task"])
            labels.append(item["label"])

        max_length = self.seq_max_length

        def pad_embeddings(embeddings):
            return torch.stack([
                torch.nn.functional.pad(embedding, (0, 0, 0, max_length - embedding.size(0)), "constant", 0)
                for embedding in embeddings
            ])

        epitope_embeddings = pad_embeddings(epitope_embeddings)
        tra_cdr3_embeddings = pad_embeddings(tra_cdr3_embeddings)
        trb_cdr3_embeddings = pad_embeddings(trb_cdr3_embeddings)

        labels = torch.stack(labels)

        return {
            "epitope_embedding": epitope_embeddings,
            "tra_cdr3_embedding": tra_cdr3_embeddings,
            "trb_cdr3_embedding": trb_cdr3_embeddings,
            "v_alpha": v_alpha,
            "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": labels
        }

def main():
    experiment_name = f"Experiment Evaluation - {MODEL_NAME}"
    load_dotenv()
    PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
    print(f"PROJECT_NAME: {PROJECT_NAME}")
    run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba-zhaw")
    config = wandb.config

    # Download corresponding artifact (= dataset) from W&B
    precision = "allele"  # or allele
    dataset_name = f"paired_{precision}"
    artifact = run.use_artifact(f"{dataset_name}:latest")
    data_dir = artifact.download(f"./WnB_Experiments_Datasets/paired_{precision}")

    train_file_path = f"{data_dir}/{precision}/train.tsv"
    test_file_path = f"{data_dir}/{precision}/test.tsv"
    unseen_test_file_path = f"{data_dir}/{precision}/test_reclassified_paired_specific.tsv"
    val_file_path = f"{data_dir}/{precision}/validation.tsv"

    df_train = pd.read_csv(train_file_path, sep="\t")
    df_test = pd.read_csv(test_file_path, sep="\t")
    df_unseen_test = pd.read_csv(unseen_test_file_path, sep="\t")
    df_val = pd.read_csv(val_file_path, sep="\t")
    df_full = pd.concat([df_train, df_test, df_unseen_test, df_val])

    traV_dict = column_to_dictionray(df_full, "TRAV")
    traJ_dict = column_to_dictionray(df_full, "TRAJ")
    trbV_dict = column_to_dictionray(df_full, "TRBV")
    trbJ_dict = column_to_dictionray(df_full, "TRBJ")
    mhc_dict = column_to_dictionray(df_full, "MHC")

    traV_embed_len = get_embed_len(df_full, "TRAV")
    traJ_embed_len = get_embed_len(df_full, "TRAJ")
    trbV_embed_len = get_embed_len(df_full, "TRBV")
    trbJ_embed_len = get_embed_len(df_full, "TRBJ")
    mhc_embed_len = get_embed_len(df_full, "MHC")

    embed_base_dir = "/teamspace/studios/this_studio/BA/paired"

    unseen_test_dataset = PairedVanilla(unseen_test_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)

    # can be seen in the W&B log
    SEQ_MAX_LENGTH = 30
    print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")
    pad_collate = PadCollate(SEQ_MAX_LENGTH).pad_collate

    generator = torch.Generator().manual_seed(42)
    test_sampler = SequentialSampler(unseen_test_dataset)

    test_dataloader = DataLoader(
        unseen_test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    trainer = pl.Trainer(accelerator=DEVICE)

    hyperparameters = {}
    hyperparameters["optimizer"] = "adam"
    hyperparameters["learning_rate"] = 5e-3
    hyperparameters["weight_decay"] = 0.075
    hyperparameters["dropout_attention"] = 0.3
    hyperparameters["dropout_linear"] = 0.45

    model = VanillaModel(EMBEDDING_SIZE, SEQ_MAX_LENGTH, DEVICE, traV_embed_len, traJ_embed_len, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters)

    checkpoint_path = "/teamspace/studios/this_studio/BA_ZHAW/models/vanilla/checkpoints/resilient-sweep-17/epoch=16-val_loss=0.44.ckpt"
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys()) 
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer.test(model, dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
