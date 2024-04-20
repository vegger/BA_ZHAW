import os
# import wandb
import torch
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from modelV1 import ModelV1
from modelV2 import ModelV2
from paired import Paired  
# from dotenv import load_dotenv


torch.manual_seed(42)

# ---------------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------------
# NOT USED YET -----
LEARNING_RATE = 5e-3
BETA_1 = 0.9
BETA_2 = 0.98
EPSILON = 1e-9
# NOT USED YET -----


EMBEDDING_SIZE = 1024
BATCH_SIZE = 32
EPOCHS = 10
# change to: NUM_WORKERS = 0 for debugging!!!
# Furthermore by increasing it gets sooooo slow... may be because of the pre-processing
NUM_WORKERS = 0

# helper vars
MODEL_OUT = 'modelV1.pth'
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class PadCollate:
    def __init__(self, seq_max_length):
        self.seq_max_length = seq_max_length

    def pad_collate(self, batch):
        epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings, labels = [], [], [], []
        for item in batch:
            epitope_embeddings.append(item["epitope_embedding"])
            tra_cdr3_embeddings.append(item["tra_cdr3_embedding"])
            trb_cdr3_embeddings.append(item["trb_cdr3_embedding"])
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
            "label": labels
        }


'''
def setup(experiment_name: str, ): 
    load_dotenv()
    PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
    experiment_description = experiment_name
    wandb.init(project=PROJECT_NAME, job_type=f"{experiment_description}")
    wandb.init()
'''

def main(): 
    # experiment_name = "Experiment 1 - Test"
    # setup(experiment_name=experiment_name)

    
    # -----------------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------------
    # Download corresponding artifact (= dataset) from W&B
    # TODO: maybe implement a main() with arguments to pass which artifact to download
    '''
    ! HERE WE NEED TO UPLOAD /DOWNLOAD THE DATASET WITHOUT (!!!!!) EMBEDDINGS !
    dataset_name = "Paired_with_Negatives"
    artifact = wandb.use_artifact(f"{dataset_name}:latest")
    BLABLABLA_table = artifact.get(f"{dataset_name}_table.table.json")
    dataset_dir = artifact.download()
    '''
    
    # Initialize dataset
    dataset = Paired()

    SEQ_MAX_LENGTH = dataset.get_max_length()
    print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")

    pad_collate = PadCollate(SEQ_MAX_LENGTH).pad_collate

    # Ratio of the datasets (TRAIN, TEST, VAL) => now 70%/15%/15%
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    print(f"train_size: {train_size}")
    val_size = int(0.15 * total_size)
    print(f"val_size: {val_size}")
    test_size = total_size - train_size - val_size
    print(f"test_size: {test_size}")


    # For reproducability
    generator1 = torch.Generator().manual_seed(42)

    train_data, remaining_data = random_split(dataset, [train_size, total_size - train_size], generator=generator1)
    val_data, test_data = random_split(remaining_data, [val_size, test_size], generator=generator1)

    train_sampler = RandomSampler(train_data)
    '''
    change them to random fixed the issue with the AUROC error, where no true positive samples got found 
    val_sampler = SequentialSampler(val_data)
    test_sampler = SequentialSampler(test_data)
    '''
    val_sampler = RandomSampler(val_data)
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        sampler=test_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=pad_collate,
    )

    # ---------------------------------------------------------------------------------
    # model 
    # ---------------------------------------------------------------------------------
    model = ModelV1(BATCH_SIZE, EMBEDDING_SIZE, SEQ_MAX_LENGTH)
    # model = ModelV2(SEQ_MAX_LENGTH)

    # ---------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------
    # Initialize loggers
    # wandb_logger = WandbLogger(project="YourProjectName")
    tensorboard_logger = TensorBoardLogger("tb_logs", name="ModelV1")

    # Initialize a simple model checkpoint callback to save your models
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./model_checkpoints',
        filename='ModelV2-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        # logger=[wandb_logger, tensorboard_logger],
        logger=[tensorboard_logger],
        # callbacks=[checkpoint_callback],
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Adjust based on your setup
        accelerator="gpu",
        log_every_n_steps=1,
    )

    # Training
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Testing
    # TODO maybe create 2 different trainers to be able to test devices=1 in the trainer for the test: 
    # Ref: https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
    test_RES = trainer.test(model, dataloaders=test_dataloader)
    print(f"test_RES: {test_RES}")
    validate_RES = trainer.validate(model, dataloaders=val_dataloader)
    print(f"validate_RES: {validate_RES}")

    # Close W&B run
    # wandb_logger.experiment.finish()

    # ---------------------------------------------------------------------------------
    # save model
    # ---------------------------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved PyTorch Model State to {MODEL_OUT}")
    


if __name__ == '__main__':
    main()
