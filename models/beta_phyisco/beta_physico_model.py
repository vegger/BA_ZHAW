import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, output_attn_w=False, n_hidden=64, dropout=0.1):
        """
        Args:
          embed_dim: Number of input and output features.
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block.
          dropout: Dropout rate after the first layer of the MLP and in two places on the main path (before
                   combining the main path with a skip connection).
        """
        super(TransformerBlock,self).__init__()
        self.mh = nn.MultiheadAttention(embed_dim,n_heads,dropout=dropout)
        self.drop1= nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop2= nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim,n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden,embed_dim))
        self.output_attn_w = output_attn_w

    def forward(self, x):
        """
        Args:
          x of shape (max_seq_length, batch_size, embed_dim): Input sequences.
          
        Returns:
          xm of shape (max_seq_length, batch_size, embed_dim): Encoded input sequences.
          attn_w of shape (batch_size, max_seq_length, max_seq_length)
        """
        xm, attn_w= self.mh(x,x,x)
        xm = self.drop1(xm)
        xm = self.norm1(x+xm)
        x = self.ff(xm)
        x = self.drop2(x)
        xm = self.norm2(x+xm)
        return (xm,attn_w) if self.output_attn_w else xm
    

class Classifier(nn.Module): 
    # hidden_dim: 64
    def __init__(self, input_dim, hidden_dim, dropout_linear):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU()
        self.downsampling_linear = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.res_block1 = ResidualBlock(hidden_dim, dropout_linear)
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.downsampling_linear(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.res_block1(x)
        x = self.final_layer(x)
        return x
    

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_linear):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_linear)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) 

    def forward(self, x):
        residual = x
        out = self.bn1(x) 
        out = self.relu(out)
        out = self.linear1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)  
        out = self.linear2(out)
        out += residual 
        return out


class MLP_Physico(nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_linear): 
        super(MLP_Physico, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_linear)
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.final_layer(x)
        return x


class BetaPhysicoModel(pl.LightningModule):
    def __init__(self, embed_dim, max_seq_length, device_, trbV_embed_len, trbJ_embed_len, mhc_embed_len, hyperparameters):
        super(BetaPhysicoModel, self).__init__()
        """
        This model uses beta only input!
        Furthermore, this model uses physicochemical properties.        
        """
        self.save_hyperparameters()
        # for evaluation of the model
        self.auroc = BinaryAUROC(thresholds=None)
        self.avg_precision = BinaryAveragePrecision(thresholds=None)
        self.hyperparameters = hyperparameters
        self.device_ = device_
        self.max_seq_length = max_seq_length

        self.test_predictions = []
        self.test_labels = []
        self.test_tasks = []
        self.epitopes = []
        self.trb_cdr3s = []

        
        self.allele_info_dim = 1024
        self.trbV_embed = nn.Embedding(trbV_embed_len, self.allele_info_dim)
        self.trbJ_embed = nn.Embedding(trbJ_embed_len, self.allele_info_dim)
        self.mhc_embed = nn.Embedding(mhc_embed_len, self.allele_info_dim)
        
        # Define MLP for Physicos
        self.physico_in_dim = 101
        self.physico_hidden_dim = 64
        self.physico_output_dim = embed_dim
        self.mlp_physico_epitope = MLP_Physico(self.physico_in_dim, self.physico_hidden_dim, self.physico_output_dim, self.hyperparameters["dropout_linear"])
        self.mlp_physico_trb = MLP_Physico(self.physico_in_dim, self.physico_hidden_dim, self.physico_output_dim, self.hyperparameters["dropout_linear"])

        # Define TransformerBlock for TRB+Epitope after physico attention
        # Note: embed_dim must be dividable by num_heads!
        self.transformer_in = embed_dim
        self.num_heads = 4
        # same as in EPIC-TRACE paper
        self.n_hidden = int(1.5*self.transformer_in)
        self.multihead_attn_global = TransformerBlock(self.transformer_in, self.num_heads, False, self.n_hidden, self.hyperparameters["dropout_attention"])

        # Define Classifier
        # flattened output of the transformer:
        # 2*(2*(self.max_seq_length+1)+3)*self.embed_dim
        self.classifier_hidden = 64
        self.classifier_in = (2*(max_seq_length+1)+3)*embed_dim
        self.classifier = Classifier(self.classifier_in, self.classifier_hidden, self.hyperparameters["dropout_linear"])
    

    def forward(self, epitope, trb_cdr3, epitope_physico, trb_physico, v_beta, j_beta, mhc):
        '''
        print(f"epitope.shape: {epitope.shape}")
        print(f"trb_cdr3.shape: {trb_cdr3.shape}")
        
        print(f"len(v_beta): {len(v_beta)}")
        print(f"len(j_beta): {len(j_beta)}")
        print(f"len(mhc): {len(mhc)}")
        print(f"type(v_beta): {type(v_beta)}")
        print(f"type(j_beta): {type(j_beta)}")
        print(f"type(mhc): {type(mhc)}")
        '''

        epitope_physico_embedd = self.mlp_physico_epitope(epitope_physico).unsqueeze(1)
        trb_physico_embedd = self.mlp_physico_trb(trb_physico).unsqueeze(1)

        epitope_with_physico = torch.cat([epitope, epitope_physico_embedd], dim=1)
        trb_with_physico = torch.cat([trb_cdr3, trb_physico_embedd], dim=1)

        trb_epitope = torch.cat([trb_with_physico, epitope_with_physico], dim=1)
        # print(f"trb_epitope.shape: {trb_epitope.shape}")

        trb_v_embed = self.trbV_embed(v_beta.to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        trb_j_embed = self.trbJ_embed(j_beta.to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        mhc_embed = self.mhc_embed(mhc.to(self.device_)).unsqueeze(0).permute(1, 0, 2)
        
        '''
        print(f"trb_v_embed: {trb_v_embed.shape}")
        print(f"trb_j_embed: {trb_j_embed.shape}")
        print(f"mhc_embed: {mhc_embed.shape}")
        '''

        trb_epitope_vj_mhc = torch.cat([trb_epitope, trb_v_embed, trb_j_embed, mhc_embed], dim=1)
        # print(f"trb_epitope_vj_mhc.shape: {trb_epitope_vj_mhc.shape}")

        trb_epitope_vj_mhc_attention = self.multihead_attn_global(trb_epitope_vj_mhc) 
        # print(f"trb_epitope_vj_mhc_attention.shape: {trb_epitope_vj_mhc_attention.shape}")       

        trb_epitope_vj_mhc_flatten = trb_epitope_vj_mhc_attention.view(trb_epitope_vj_mhc_attention.size(0), -1)
        # print(f"trb_epitope_vj_mhc_flatten.shape: {trb_epitope_vj_mhc_flatten.shape}")
        
        logits = self.classifier(trb_epitope_vj_mhc_flatten)
        # print(f"logits: {logits}")
        return logits
    

    def training_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        epitope_physico = batch["epitope_physico"]
        trb_pyhsico = batch["trb_physico"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        label = batch["label"]
        
        output = self(epitope_embedding, trb_cdr3_embedding, epitope_physico, trb_pyhsico, v_beta, j_beta, mhc).squeeze(1)

        # print(f"this is y in training_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("train_loss", loss, on_step=True, prog_bar=True, batch_size=len(batch))
        return loss


    def test_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        epitope_sequence = batch["epitope_sequence"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        trb_cdr3_sequence = batch["trb_cdr3_sequence"]
        epitope_physico = batch["epitope_physico"]
        trb_pyhsico = batch["trb_physico"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        task = batch["task"]
        label = batch["label"]
        
        output = self(epitope_embedding, trb_cdr3_embedding, epitope_physico, trb_pyhsico, v_beta, j_beta, mhc).squeeze(1)
        prediction = torch.sigmoid(output)
        
        self.test_predictions.append(prediction)
        self.test_labels.append(label)
        self.test_tasks.append(task[0])
        self.epitopes.append(epitope_sequence)
        self.trb_cdr3s.append(trb_cdr3_sequence)

        test_loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("test_loss", test_loss, batch_size=len(batch))
        
        return test_loss, prediction, label
    

    def on_test_epoch_end(self):
        test_predictions = torch.stack(self.test_predictions)
        test_labels = torch.stack(self.test_labels)
        # print(f"on_test_epoch_end, test_labels: {test_labels}")
        test_tasks = self.test_tasks
    
        tpp_1 = []
        tpp_2 = []
        tpp_3 = []
        tpp_4 = []

        for i, task in enumerate(test_tasks):
            if task == "TPP1": 
                # print(f"task 1: {task}")
                tpp_1.append((test_predictions[i], test_labels[i]))
            elif task == "TPP2": 
                # print(f"task 2: {task}")
                tpp_2.append((test_predictions[i], test_labels[i]))
            elif task == "TPP3": 
                # print(f"task 3: {task}")
                tpp_3.append((test_predictions[i], test_labels[i]))
            elif task == "TPP4":
                # print("in TPP4")
                tpp_4.append((test_predictions[i], test_labels[i]))
            else: 
                print("ERROR IN TASK")
 
        self.log("ROCAUC_Test_global", self.auroc(test_predictions, test_labels), prog_bar=True)
        self.log("AP_Test_global", self.avg_precision(test_predictions, test_labels.to(torch.long)), prog_bar=True)

        self.log("ROCAUC_Test_TPP1", self.auroc(torch.tensor([item[0] for item in tpp_1]), torch.tensor([item[1] for item in tpp_1]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP1", self.avg_precision(torch.tensor([item[0] for item in tpp_1]), torch.tensor([item[1] for item in tpp_1]).to(torch.long)), prog_bar=True) 
        
        self.log("ROCAUC_Test_TPP2", self.auroc(torch.tensor([item[0] for item in tpp_2]), torch.tensor([item[1] for item in tpp_2]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP2", self.avg_precision(torch.tensor([item[0] for item in tpp_2]), torch.tensor([item[1] for item in tpp_2]).to(torch.long)), prog_bar=True)          
        
        self.log("ROCAUC_Test_TPP3", self.auroc(torch.tensor([item[0] for item in tpp_3]), torch.tensor([item[1] for item in tpp_3]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP3", self.avg_precision(torch.tensor([item[0] for item in tpp_3]), torch.tensor([item[1] for item in tpp_3]).to(torch.long)), prog_bar=True)  

        # print(f"len(tpp_4): {tpp_4}")
        self.log("ROCAUC_Test_TPP4", self.auroc(torch.tensor([item[0] for item in tpp_4]), torch.tensor([item[1] for item in tpp_4]).to(torch.long)), prog_bar=True)
        self.log("AP_Test_TPP4", self.avg_precision(torch.tensor([item[0] for item in tpp_4]), torch.tensor([item[1] for item in tpp_4]).to(torch.long)), prog_bar=True)  

        test_predictions = torch.stack(self.test_predictions).squeeze(1).cpu().numpy()  
        test_labels = torch.stack(self.test_labels).squeeze(1).cpu().numpy()            
        test_tasks = np.array(self.test_tasks)
        test_epitopes = np.array(self.epitopes).squeeze(1)
        test_trb_cdr3s = np.array(self.trb_cdr3s).squeeze(1)

        
        print(f"test_predictions.shape: {test_predictions.shape}")  
        print(f"test_labels.shape: {test_labels.shape}")   
        print(f"test_tasks.shape: {test_tasks.shape}") 
        print(f"test_epitopes: {test_epitopes.shape}")           
        print(f"test_trb_cdr3s: {test_trb_cdr3s.shape}")                       
        

        data = {
            "test_epitopes": test_epitopes, 
            "test_trb_cdr3s": test_trb_cdr3s,
            "test_predictions": test_predictions,
            "test_labels": test_labels,
            "test_tasks": test_tasks, 
        }

        df = pd.DataFrame(data)
        # df.to_csv("./test_physico_beta_df.tsv", sep="\t")

        self.test_predictions.clear()
        self.test_labels.clear()
        self.test_tasks.clear()


    def validation_step(self, batch, batch_idx):
        epitope_embedding = batch["epitope_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        epitope_physico = batch["epitope_physico"]
        trb_pyhsico = batch["trb_physico"]
        v_beta = batch["v_beta"]
        j_beta = batch["j_beta"]
        mhc = batch["mhc"]
        label = batch["label"]
        
        output = self(epitope_embedding, trb_cdr3_embedding, epitope_physico, trb_pyhsico, v_beta, j_beta, mhc).squeeze(1)
        
        val_loss = F.binary_cross_entropy_with_logits(output, label)
        self.log("val_loss", val_loss, batch_size=len(batch))
        
        prediction = torch.sigmoid(output)
        self.log("ROCAUC_Val", self.auroc(prediction, label), on_epoch=True, prog_bar=True, batch_size=len(batch))
        self.log("AP_Val", self.avg_precision(prediction, label.to(torch.long)), on_epoch=True, prog_bar=True, batch_size=len(batch))

        return val_loss
    
    
    def configure_optimizers(self):
        optimizer = self.hyperparameters["optimizer"]
        learning_rate = self.hyperparameters["learning_rate"]
        weight_decay = self.hyperparameters["weight_decay"]
        betas = (0.9, 0.98)
        
        if optimizer == "sgd": 
            optimizer = torch.optim.SGD(self.parameters(),
                        lr=learning_rate, momentum=0.9)        
        elif optimizer == "adam": 
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        else: 
            print("OPTIMIZER NOT FOUND")
        
        return optimizer
