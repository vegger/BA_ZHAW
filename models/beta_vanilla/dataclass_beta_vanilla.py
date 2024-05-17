import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class BetaVanilla(Dataset):
    def __init__(self, dataset_path, embed_base_path, trbV, trbJ, mhc, transform=None):
        """
        Dataclass for the experiment NOT USING (!) physicochemical properties.
        """
        
        self.dataset_path = dataset_path
        self.epitope_embeddings_path = f"{embed_base_path}/Epitope_beta_embeddings.npz"
        # self.tra_embeddings_path = f"{embed_base_path}/TRA_CDR3_embeddings.npz"
        self.trb_embeddings_path = f"{embed_base_path}/TRB_beta_embeddings.npz"
        
        self.transform = transform
        self.data_frame = pd.read_csv(self.dataset_path, sep='\t')
        
        epitope_embeddings = np.load(self.epitope_embeddings_path)
        # tra_embeddings = np.load(self.tra_embeddings_path)
        trb_embeddings = np.load(self.trb_embeddings_path)

        # traV_dict = traV
        # traJ_dict = traJ
        trbV_dict = trbV
        trbJ_dict = trbJ
        mhc_dict = mhc
        
        self.data_frame["Epitope Embedding"] = self.data_frame["Epitope"].map(epitope_embeddings)
        # self.data_frame["TRA_CDR3 Embedding"] = self.data_frame["TRA_CDR3"].map(tra_embeddings)
        self.data_frame["TRB_CDR3 Embedding"] = self.data_frame["TRB_CDR3"].map(trb_embeddings)

        # self.data_frame["TRAV Index"] = self.data_frame["TRAV"].map(traV_dict)
        # self.data_frame["TRAJ Index"] = self.data_frame["TRAJ"].map(traJ_dict)
        self.data_frame["TRBV Index"] = self.data_frame["TRBV"].map(trbV_dict)
        self.data_frame["TRBJ Index"] = self.data_frame["TRBJ"].map(trbJ_dict)
        self.data_frame["MHC Index"] = self.data_frame["MHC"].map(mhc_dict)

        columns = list(self.data_frame.columns) 
        columns.remove('Binding')
        columns.append('Binding')
        self.data_frame = self.data_frame[columns]
        

    def __len__(self):
        return len(self.data_frame)

    
    def __getitem__(self, index):
        row = self.data_frame.iloc[index]  

        epitope_embedding = torch.tensor(row["Epitope Embedding"], dtype=torch.float32)
        # tra_cdr3_embedding = torch.tensor(row["TRA_CDR3 Embedding"], dtype=torch.float32)
        trb_cdr3_embedding = torch.tensor(row["TRB_CDR3 Embedding"], dtype=torch.float32)
        # v_alpha = row["TRAV Index"]
        # j_alpha = row["TRAJ Index"]
        v_beta = row["TRBV Index"]
        j_beta = row["TRBJ Index"]
        mhc = row["MHC Index"]
        task = row["task"]
        
        label = torch.tensor(row["Binding"], dtype=torch.float)

        batch = {
            "epitope_embedding": epitope_embedding,
            # "tra_cdr3_embedding": tra_cdr3_embedding,
            "trb_cdr3_embedding": trb_cdr3_embedding,
            # "v_alpha": v_alpha,
            # "j_alpha": j_alpha,
            "v_beta": v_beta,
            "j_beta": j_beta,
            "mhc": mhc,
            "task": task,
            "label": label,
        }

        if self.transform:
            batch = self.transform(batch)

        return batch
    

    def get_max_length(self):
        df = self.data_frame
        df = df[["Epitope Embedding", "TRB_CDR3 Embedding"]]
        epitope_embeddings = []
        tra_cdr3_embeddings = []
        trb_cdr3_embeddings = []

        for i, row in df.iterrows():
            epitope_embeddings.append(row["Epitope Embedding"])
            # tra_cdr3_embeddings.append(row["TRA_CDR3 Embedding"])
            trb_cdr3_embeddings.append(row["TRB_CDR3 Embedding"])

        max_length = max(
            max(embedding.shape[0] for embedding in epitope_embeddings),
            # max(embedding.shape[0] for embedding in tra_cdr3_embeddings),
            max(embedding.shape[0] for embedding in trb_cdr3_embeddings)
        )

        return max_length
        