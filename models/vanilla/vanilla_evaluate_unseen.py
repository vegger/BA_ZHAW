from models.vanilla.vanilla_model import VanillaModel
from models.vanilla.dataclass_paired_vanilla import PairedVanilla


EMBEDDING_SIZE = 1024
BATCH_SIZE = 128

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Download corresponding artifact (= dataset) from W&B
precision = "gene" # or allele
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

embed_base_dir = "/teamspace/studios/this_studio/BA/paired"


train_dataset = PairedVanilla(train_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)
test_dataset = PairedVanilla(test_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)
val_dataset = PairedVanilla(val_file_path, embed_base_dir, traV_dict, traJ_dict, trbV_dict, trbJ_dict, mhc_dict)

SEQ_MAX_LENGTH = max(train_dataset.get_max_length(), test_dataset.get_max_length(), val_dataset.get_max_length())
print(f"this is SEQ_MAX_LENGTH: {SEQ_MAX_LENGTH}")

# not important which values as model from checkpoint and only used in .eval mode!
hyperparameters = {}
hyperparameters["optimizer"] = "adam"
hyperparameters["learning_rate"] = 5e-3
hyperparameters["weight_decay"] = 0.075
hyperparameters["dropout_attention"] = 0.3
hyperparameters["dropout_linear"] = 0.45

model = VanillaModel(EMBEDDING_SIZE, SEQ_MAX_LENGTH, DEVICE, traV_embed_len, traJ_embed_len, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters)
