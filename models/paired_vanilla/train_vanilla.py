import os
import wandb
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging
from vanilla_model import VanillaModel
from dataclass_paired_vanilla import PairedVanilla 
from dotenv import load_dotenv


torch.manual_seed(42)

# ---------------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------------
MODEL_NAME = "VanillaModel"
EMBEDDING_SIZE = 1024
BATCH_SIZE = 128
EPOCHS = 250
# IMPORTANT: keep NUM_WORKERS = 0!
NUM_WORKERS = 0

MODEL_OUT = f"{MODEL_NAME}.pth"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def set_hyperparameters(config):
    hyperparameters = {}
    hyperparameters["optimizer"] = config.optimizer
    hyperparameters["learning_rate"] = config.learning_rate
    hyperparameters["weight_decay"] = config.weight_decay
    hyperparameters["dropout_attention"] = config.dropout_attention
    hyperparameters["dropout_linear"] = config.dropout_linear

    return hyperparameters


class PadCollate:
    def __init__(self, seq_max_length):
        self.seq_max_length = seq_max_length

    def pad_collate(self, batch):
        epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings = [], [], []
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

        labels = torch.stack(labels)

        return {
            "epitope_embedding": epitope_embeddings,
            "epitope_sequence": epitope_sequence,
            "tra_cdr3_embedding": tra_cdr3_embeddings,
            "tra_cdr3_sequence": tra_cdr3_sequence,
            "trb_cdr3_embedding": trb_cdr3_embeddings,
            "trb_cdr3_sequence": trb_cdr3_sequence,
            "v_alpha": v_alpha,
            "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": labels
        }


def column_to_dictionray(df, column_name): 
    list_of_column = df[column_name].unique()
    dictionary = {}
    for index, item in enumerate(list_of_column): 
        # print(f"index: {index}, item: {item}")
        dictionary[item] = index
    
    return dictionary


def get_embed_len(df, column_name): 
    list_of_column = df[column_name].unique()    
    return len(list_of_column)


def main():
    precision = "allele" # or gene
    embed_base_dir = f"../../data/embeddings/paired/{precision}"
    hyperparameter_tuning_with_WnB = False

    # -----------------------------------------------------------------------------
    # W&B Setup
    # -----------------------------------------------------------------------------

    experiment_name = f"Experiment - {MODEL_NAME}"
    load_dotenv()
    PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
    print(f"PROJECT_NAME: {PROJECT_NAME}")
    run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="ba-zhaw")
    config = wandb.config

    # -----------------------------------------------------------------------------
    # data (from W&B)
    # -----------------------------------------------------------------------------
    # Download corresponding artifact (= dataset) from W&B
    dataset_name = f"paired_{precision}"
    artifact = run.use_artifact(f"{dataset_name}:latest")
    data_dir = artifact.download(f"./WnB_Experiments_Datasets/paired_{precision}")
    
    train_file_path = f"{data_dir}/{precision}/train.tsv"
    test_file_path = f"{data_dir}/{precision}/test.tsv"
    val_file_path = f"{data_dir}/{precision}/validation.tsv"

    df_train = pd.read_csv(train_file_path, sep="\t")
    df_test = pd.read_csv(test_file_path, sep="\t")
    df_val = pd.read_csv(val_file_path, sep="\t")
    df_full = pd.concat([df_train, df_test, df_val])
    
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

    train_dataset = PairedVanilla(train_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)
    test_dataset = PairedVanilla(test_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)
    val_dataset = PairedVanilla(val_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)

    SEQ_MAX_LENGTH = max(train_dataset.get_max_length(), test_dataset.get_max_length(), val_dataset.get_max_length())
    print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")

    pad_collate = PadCollate(SEQ_MAX_LENGTH).pad_collate

    # For reproducability
    generator = torch.Generator().manual_seed(42)
    train_sampler = RandomSampler(train_dataset, generator=generator)
    val_sampler = RandomSampler(val_dataset, generator=generator)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,

    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )
    # ---------------------------------------------------------------------------------
    # model 
    # ---------------------------------------------------------------------------------
    if hyperparameter_tuning_with_WnB:
        hyperparameters = set_hyperparameters(config)
    else:
        hyperparameters = {}
        hyperparameters["optimizer"] = "adam"
        hyperparameters["learning_rate"] = 5e-3
        hyperparameters["weight_decay"] = 0.075
        hyperparameters["dropout_attention"] = 0.3
        hyperparameters["dropout_linear"] = 0.45
    
    model = VanillaModel(EMBEDDING_SIZE, SEQ_MAX_LENGTH, DEVICE, traV_embed_len, traJ_embed_len, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters)
    # ---------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------
    # Initialize loggers
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=experiment_name)
    # This logs gradients
    wandb_logger.watch(model)
    tensorboard_logger = TensorBoardLogger("tb_logs", name=f"{MODEL_NAME}")

    # Callbacks
    run_name = wandb.run.name  
    checkpoint_dir = f"checkpoints/{run_name}"
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        monitor="AP_Val",  
        mode="max",
        save_top_k=1  
    )

    early_stopping = EarlyStopping(
        monitor="AP_Val",  
        patience=5,        
        verbose=True,
        mode="max"        
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    swa = StochasticWeightAveraging(swa_lrs=hyperparameters["learning_rate"]*0.1, swa_epoch_start=45)

    # Training
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[model_checkpoint, early_stopping, lr_monitor, swa],  
        accelerator="gpu",
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_model_path = model_checkpoint.best_model_path
    print(f"Best model saved at {best_model_path}")
    # Testing
    test_RES = trainer.test(model, dataloaders=test_dataloader)
    print(f"test_RES: {test_RES}")
    validate_RES = trainer.validate(model, dataloaders=val_dataloader)
    print(f"validate_RES: {validate_RES}")
    # Close W&B run
    wandb_logger.experiment.finish()
    # ---------------------------------------------------------------------------------
    # save model
    # ---------------------------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved PyTorch Model State to {MODEL_OUT}")


if __name__ == '__main__':
    main()

