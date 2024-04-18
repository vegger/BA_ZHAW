import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Paired(Dataset):
    def __init__(self, transform=None):
        """
        Initialize dataset paths and optional transformations.
        """
        self.file_path = "./Dataclasses/no_Stitchr/samples.tsv"
        self.epitope_embeddings_path = "../data/Embeddings/paired/paired_epitope_embeddings.npz"
        self.tra_embeddings_path = "../data/Embeddings/paired/paired_TRA_CDR3_embeddings.npz"
        self.trb_embeddings_path = "../data/Embeddings/paired/paired_TRB_CDR3_embeddings.npz"
        self.transform = transform

        # Load the entire DataFrame
        self.data_frame = pd.read_csv(self.file_path, sep='\t')
        
        # Memory-map the embeddings
        epitope_embeddings = np.load(self.epitope_embeddings_path)
        tra_embeddings = np.load(self.tra_embeddings_path)
        trb_embeddings = np.load(self.trb_embeddings_path)
        
        self.data_frame["Epitope Embedding"] = self.data_frame["Epitope"].map(epitope_embeddings)
        self.data_frame["TRA_CDR3 Embedding"] = self.data_frame["TRA_CDR3"].map(tra_embeddings)
        self.data_frame["TRB_CDR3 Embedding"] = self.data_frame["TRB_CDR3"].map(trb_embeddings)

        columns = list(self.data_frame.columns)  # Get the list of all column names
        columns.remove('Binding')
        columns.append('Binding')
        self.data_frame = self.data_frame[columns]


    def __len__(self):
        return len(self.data_frame)

    
    def __getitem__(self, index):
        # Get the specific row for the index
        row = self.data_frame.iloc[index]  

        epitope_embedding = torch.tensor(row["Epitope Embedding"], dtype=torch.float32).detach()
        tra_cdr3_embedding = torch.tensor(row["TRA_CDR3 Embedding"], dtype=torch.float32).detach()
        trb_cdr3_embedding = torch.tensor(row["TRB_CDR3 Embedding"], dtype=torch.float32).detach()
    
        label = torch.tensor(row["Binding"], dtype=torch.float).detach()

        batch = {
            "epitope_embedding": epitope_embedding,
            "tra_cdr3_embedding": tra_cdr3_embedding,
            "trb_cdr3_embedding": trb_cdr3_embedding,
            "label": label,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch
    

    def get_max_length(self):
        df = self.data_frame
        df = df[["Epitope Embedding", "TRA_CDR3 Embedding", "TRB_CDR3 Embedding"]]
        epitope_embeddings = []
        tra_cdr3_embeddings = []
        trb_cdr3_embeddings = []

        for _, row in df.iterrows():
            epitope_embeddings.append(row["Epitope Embedding"])
            tra_cdr3_embeddings.append(row["TRA_CDR3 Embedding"])
            trb_cdr3_embeddings.append(row["TRB_CDR3 Embedding"])

        max_length = max(
            max(embedding.shape[0] for embedding in epitope_embeddings),
            max(embedding.shape[0] for embedding in tra_cdr3_embeddings),
            max(embedding.shape[0] for embedding in trb_cdr3_embeddings)
        )

        return max_length
    

if __name__ == "__main__":
        dataset = Paired()

        # print(type(dataset[0]["Epitope Embedding"]))
        print(dataset.get_max_length())
