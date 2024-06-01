import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import matplotlib.pyplot as plt
import wandb
import os
from physico_model import PhysicoModel
from dataclass_paired_physico import PairedPhysico
from dotenv import load_dotenv

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

MODEL_NAME = "PhyiscoModel"

def column_to_dictionary(df, column_name):
    list_of_column = df[column_name].unique()
    dictionary = {item: index for index, item in enumerate(list_of_column)}
    return dictionary

def get_embed_len(df, column_name):
    return len(df[column_name].unique())

class PadCollate:
    def __init__(self, seq_max_length):
        self.seq_max_length = seq_max_length

    def pad_collate(self, batch):
        epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings = [], [], []
        epitope_physico, tra_physico, trb_physico = [], [], []
        v_alpha, j_alpha, v_beta, j_beta = [], [], [], []
        epitope_sequence, tra_cdr3_sequence, trb_cdr3_sequence = [], [], []
        mhc = []
        task = []
        labels = []

        for item in batch:
            epitope_embeddings.append(item["epitope_embedding"])
            epitope_sequence.append(item["epitope_sequence"])
            tra_cdr3_embeddings.append(item["tra_cdr3_embedding"])
            tra_cdr3_sequence.append(item["tra_cdr3_sequence"])
            trb_cdr3_embeddings.append(item["trb_cdr3_embedding"])
            trb_cdr3_sequence.append(item["trb_cdr3_sequence"])
            epitope_physico.append(item["epitope_physico"])
            tra_physico.append(item["tra_physico"])
            trb_physico.append(item["trb_physico"])
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

        v_alpha = torch.tensor(v_alpha, dtype=torch.int32)
        j_alpha = torch.tensor(j_alpha, dtype=torch.int32)
        v_beta = torch.tensor(v_beta, dtype=torch.int32)
        j_beta = torch.tensor(j_beta, dtype=torch.int32)
        mhc = torch.tensor(mhc, dtype=torch.int32)

        epitope_physico = torch.stack(epitope_physico)
        tra_physico = torch.stack(tra_physico)
        trb_physico = torch.stack(trb_physico)

        labels = torch.stack(labels)

        return {
            "epitope_embedding": epitope_embeddings,
            "epitope_sequence": epitope_sequence,
            "tra_cdr3_embedding": tra_cdr3_embeddings,
            "tra_cdr3_sequence": tra_cdr3_sequence,
            "trb_cdr3_embedding": trb_cdr3_embeddings,
            "trb_cdr3_sequence": trb_cdr3_sequence,
            "epitope_physico": epitope_physico, 
            "tra_physico": tra_physico,
            "trb_physico": trb_physico,
            "v_alpha": v_alpha,
            "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": labels
        }


def forward_with_softmax(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding, v_alpha, j_alpha, v_beta, j_beta, mhc):
    output = model(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding, v_alpha, j_alpha, v_beta, j_beta, mhc)
    print(f"output in forward_with_softmax(): {output}")
    prinf(f"F.sigmoid(output): {F.sigmoid(output)}")
    return F.sigmoid(output)


def main():
    global model

    experiment_name = f"Experiment Evaluation (Allele) - {MODEL_NAME}"
    load_dotenv()
    PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
    print(f"PROJECT_NAME: {PROJECT_NAME}")
    run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba-zhaw")
    config = wandb.config

    # Download corresponding artifact (= dataset) from W&B
    precision = "allele"  # or gene
    # precision = "gene"
    print(f"precision: {precision}")
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

    traV_dict = column_to_dictionary(df_full, "TRAV")
    traJ_dict = column_to_dictionary(df_full, "TRAJ")
    trbV_dict = column_to_dictionary(df_full, "TRBV")
    trbJ_dict = column_to_dictionary(df_full, "TRBJ")
    mhc_dict = column_to_dictionary(df_full, "MHC")

    traV_embed_len = get_embed_len(df_full, "TRAV")
    traJ_embed_len = get_embed_len(df_full, "TRAJ")
    trbV_embed_len = get_embed_len(df_full, "TRBV")
    trbJ_embed_len = get_embed_len(df_full, "TRBJ")
    mhc_embed_len = get_embed_len(df_full, "MHC")

    embed_base_dir = "/teamspace/studios/this_studio/BA/paired"

    physico_base_dir = "/teamspace/studios/this_studio/BA/physico"

    train_physico_epi = f"{physico_base_dir}/scaled_train_paired_epitope_{precision}_physico.npz"
    train_physico_tra = f"{physico_base_dir}/scaled_train_paired_TRA_{precision}_physico.npz"
    train_physico_trb = f"{physico_base_dir}/scaled_train_paired_TRB_{precision}_physico.npz"
    test_physico_epi = f"{physico_base_dir}/scaled_test_paired_epitope_{precision}_physico.npz"
    test_physico_tra = f"{physico_base_dir}/scaled_test_paired_TRA_{precision}_physico.npz"
    test_physico_trb = f"{physico_base_dir}/scaled_test_paired_TRB_{precision}_physico.npz"
    val_physico_epi = f"{physico_base_dir}/scaled_validation_paired_epitope_{precision}_physico.npz"
    val_physico_tra = f"{physico_base_dir}/scaled_validation_paired_TRA_{precision}_physico.npz"
    val_physico_trb = f"{physico_base_dir}/scaled_validation_paired_TRB_{precision}_physico.npz"

    unseen_test_dataset = PairedPhysico(unseen_test_file_path, embed_base_dir, test_physico_epi, test_physico_tra, test_physico_trb, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)

    # can be seen in the W&B log, same for both Allele and Gene (SEQ_MAX_LENGTH = 30)
    SEQ_MAX_LENGTH = 30
    print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")
    pad_collate = PadCollate(SEQ_MAX_LENGTH).pad_collate

    generator = torch.Generator().manual_seed(42)
    test_sampler = SequentialSampler(unseen_test_dataset)
    # test_sampler = SequentialSampler(test_dataset)

    test_dataloader = DataLoader(
        unseen_test_dataset,
        # test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    trainer = pl.Trainer(accelerator=DEVICE)

    # Unimportant as model anyways in eval but necessary class variable
    hyperparameters = {}
    hyperparameters["optimizer"] = "adam"
    hyperparameters["learning_rate"] = 5e-3
    hyperparameters["weight_decay"] = 0.075
    hyperparameters["dropout_attention"] = 0.3
    hyperparameters["dropout_linear"] = 0.45

    model = PhysicoModel(EMBEDDING_SIZE, SEQ_MAX_LENGTH, DEVICE, traV_embed_len, traJ_embed_len, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters)
    # Paired Phyisco Gene: 
    # checkpoint_path = ""
    # Paired Physico Allele:
    # checkpoint_path = "/teamspace/studios/this_studio/BA_ZHAW/models/physico/checkpoints/laced-sweep-12/epoch=01-val_loss=0.69.ckpt"
    # checkpoint_path = "/teamspace/studios/this_studio/BA_ZHAW/models/physico/checkpoints/brisk-sweep-17/epoch=08-val_loss=0.69.ckpt"
    # 
    # checkpoint_path = "/teamspace/studios/this_studio/BA_ZHAW/models/physico/checkpoints/leafy-sweep-29/epoch=11-val_loss=0.69.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    print(checkpoint.keys())
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    
    trainer.test(model, dataloaders=test_dataloader) 

    '''
    # Working but not used yet in report as time is running
    model.to(DEVICE)
    model.eval()

    # TODO: here instead of "only" check influence of TRA_V => pass a list
    lig = LayerIntegratedGradients(model, [model.traV_embed])
    
    # ig = IntegratedGradients(forward_with_softmax)
    for batch in test_dataloader:
        # print(f"batch: \n{batch}")
        epitope_embedding = batch["epitope_embedding"].requires_grad_().to(DEVICE)
        tra_cdr3_embedding = batch["tra_cdr3_embedding"].requires_grad_().to(DEVICE)
        trb_cdr3_embedding = batch["trb_cdr3_embedding"].requires_grad_().to(DEVICE)
        v_alpha = batch["v_alpha"].to(DEVICE)
        j_alpha = batch["j_alpha"].to(DEVICE)
        v_beta = batch["v_beta"].to(DEVICE)
        j_beta = batch["j_beta"].to(DEVICE)
        mhc = batch["mhc"].to(DEVICE)
        labels = batch["label"]

        inputs = (epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding, v_alpha, j_alpha, v_beta, j_beta, mhc)
        print(f"inputs[0]: {inputs[0]}")
        print(f"inputs[3] aka v_alpha.dtype: {v_alpha.dtype}")

        attributions_ig, delta = lig.attribute(
            inputs=inputs, 
            # baselines=inputs,
            n_steps=1, 
            return_convergence_delta=True
            )

        # TODO: if a list is passed to the ig => list is returned => handle it correspondingly
        # attributions_ig = attributions_ig.cpu().numpy()
        
        print(f"attributions_ig: \n{attributions_ig}")
        print(f"delta: \n{delta}")

        # feature_names = ["epitope_embedding", "tra_cdr3_embedding", "trb_cdr3_embedding", "v_alpha", "j_alpha", "v_beta", "j_beta", "mhc"]
        # plot_ig_attributions(attributions_ig.squeeze(), feature_names)

        break # check only first item of the batch! (at least for now)
    '''

if __name__ == "__main__":
    main()
