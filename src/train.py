import os
# import wandb
import torch
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from modelV1 import ModelV1
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



BATCH_SIZE = 32
EPOCHS = 150
# keep NUM_WORKERS = 0 as long as debugging
NUM_WORKERS = 0

# helper vars
MODEL_OUT = 'model.pth'
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


'''
def setup(experiment_name: str, ): 
    load_dotenv()
    PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
    experiment_description = experiment_name
    wandb.init(project=PROJECT_NAME, job_type=f"{experiment_description}")
    wandb.init()
'''

# function used for the padding of the embeddings
def pad_collate(batch):
    epitope_embeddings, tra_cdr3_embeddings, trb_cdr3_embeddings, labels = [], [], [], []

    for item in batch:
        epitope_embeddings.append(item["epitope_embedding"])
        tra_cdr3_embeddings.append(item["tra_cdr3_embedding"])
        trb_cdr3_embeddings.append(item["trb_cdr3_embedding"])
        labels.append(item["label"])

    # as not fixed, manually looked up max_value
    max_length = SEQ_MAX_LENGTH

    # Function to pad embeddings to the maximum length
    def pad_embeddings(embeddings):
        print(f"type(embeddings[0]): {type(embeddings[0])}")
        # print(f"embeddings[0].shape: {embeddings[0].shape}")

        return torch.stack([
            torch.nn.functional.pad(embedding, (0, 0, 0, max_length - embedding.size(0)), "constant", 0)
            for embedding in embeddings
        ])

    # Pad and stack all embeddings and labels
    epitope_embeddings = pad_embeddings(epitope_embeddings)
    tra_cdr3_embeddings = pad_embeddings(tra_cdr3_embeddings)
    trb_cdr3_embeddings = pad_embeddings(trb_cdr3_embeddings)
    labels = torch.stack(labels)

    # print(f"this is in pad_collate: epitope_embedding:\n{epitope_embeddings.shape}")
    return {
        "epitope_embedding": epitope_embeddings,
        "tra_cdr3_embedding": tra_cdr3_embeddings,
        "trb_cdr3_embedding": trb_cdr3_embeddings,
        "label": labels
    } 


def main(): 
    global SEQ_MAX_LENGTH

    # experiment_name = "Experiment 1 - Test"
    # setup(experiment_name=experiment_name)

    
    # -----------------------------------------------------------------------------
    # data
    # -----------------------------------------------------------------------------
    # Download corresponding artifact (= dataset) from W&B
    # TODO: maybe implement a main() with arguments to pass which artifact to download
    '''
    dataset_name = "Paired_with_Negatives"
    artifact = wandb.use_artifact(f"{dataset_name}:latest")
    BLABLABLA_table = artifact.get(f"{dataset_name}_table.table.json")
    dataset_dir = artifact.download()
    '''
    
    # for testing: 
    # dataset_dir = "./samples.tsv"


    # Initialize dataset
    dataset = Paired()

    SEQ_MAX_LENGTH = dataset.get_max_length()
    print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")


    train_data, test_data = random_split(dataset, (0.8, 0.2))

    # create samplers
    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)

    # Create data loaders.
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
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
    model = ModelV1(SEQ_MAX_LENGTH)

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
        filename='ModelV1-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        # logger=[wandb_logger, tensorboard_logger],
        logger=[tensorboard_logger],
        callbacks=[checkpoint_callback],
        # accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Adjust based on your setup
        accelerator="gpu",
        log_every_n_steps=1,
    )

    # Train the model
    trainer.fit(model, train_dataloader, test_dataloader)

    # Close W&B run
    # wandb_logger.experiment.finish()

    # ---------------------------------------------------------------------------------
    # save model
    # ---------------------------------------------------------------------------------
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved PyTorch Model State to {MODEL_OUT}")
    


if __name__ == '__main__':
    main()
