import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
# TODO: import model
from Dataclasses.no_Stitchr.noStitchrDataset import NoStitchrDataset  # Define your dataset in this file
from dotenv import load_dotenv


load_dotenv()
experiment_description = "train.py test run"
PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
wandb.init(project=PROJECT_NAME, job_type="f{experiment_description}")
wandb.init()

# Download corresponding artifact (= dataset) from W&B
# TODO: maybe implement a main() with arguments to pass which artifact to download
dataset_name = "Paired_with_Negatives"
artifact = wandb.use_artifact(f"{dataset_name}:latest")
mcpastcr_table = artifact.get(f"{dataset_name}_table.table.json")
dataset_dir = artifact.download()
# Initialize dataset
dataset = NoStitchrDataset(dataset_dir)
print(f"type of dataset is: {type(dataset)}!\n")

print(f"This is dataset: {dataset}\n")

# Configuration
batch_size = 32 # maybe increase? check w/ literature
num_workers = 4


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

'''
# Initialize model
model = YourModel()

# Initialize loggers
wandb_logger = WandbLogger(project="YourProjectName")
tensorboard_logger = TensorBoardLogger("tb_logs", name="YourModelName")

# Initialize a simple model checkpoint callback to save your models
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./model_checkpoints',
    filename='your_model_name-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# Initialize the trainer
trainer = pl.Trainer(
    max_epochs=10,
    logger=[wandb_logger, tensorboard_logger],
    callbacks=[checkpoint_callback],
    gpus=1 if torch.cuda.is_available() else 0,  # Adjust based on your setup
)

# Train the model
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# Optional: Close W&B run
wandb_logger.experiment.finish()
'''
