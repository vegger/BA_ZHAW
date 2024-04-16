import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

class ModelV1(pl.LightningModule):
    def __init__(self, input_channels):
        super(ModelV1, self).__init__()
        # Define Conv1D layers for αCDR3, Epitope, βCDR3
        self.in_channels = input_channels
        self.out_channels = 3
        self.kernel_size = 3
        self.drop_out_p = 0.3
        self.embed_dim = 1024
        # must embed_dim must be dividable by num_heads!
        self.num_heads = 8
        self.in_features1 = 1024
        self.out_features1 = 1024
        self.in_features2 = 1024
        self.out_features2 = 1

        self.conv1d_alpha = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
        self.conv1d_epitope = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
        self.conv1d_beta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_out_p )

        '''
        # Define linear layers for aV, aJ, bV, bJ
        self.linear_alpha_v = nn.Linear(in_features=..., out_features=...)
        self.linear_alpha_j = nn.Linear(in_features=..., out_features=...)
        self.linear_beta_v = nn.Linear(in_features=..., out_features=...)
        self.linear_beta_j = nn.Linear(in_features=..., out_features=...)
        '''

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Reduces each feature to a single number

        self.post_attn_linear = nn.Linear(in_features=self.in_features1, out_features=self.out_features1)

        self.final_layer = nn.Linear(in_features=self.in_features2, out_features=self.out_features2)
        

    # def forward(self, tra_cdr3, epitope, trb_cdr3, alpha_v, alpha_j, beta_v, beta_j):
    def forward(self, epitope, tra_cdr3, trb_cdr3):
        # Apply conv1d + ReLU + Dropout for αCDR3, Epitope, βCDR3
        tra_cdr3 = self.dropout(self.relu(self.conv1d_alpha(tra_cdr3)))
        epitope = self.dropout(self.relu(self.conv1d_epitope(epitope)))
        trb_cdr3 = self.dropout(self.relu(self.conv1d_beta(trb_cdr3)))

        '''
        # Process aV, aJ, bV, bJ with linear layers
        alpha_v = self.linear_alpha_v(alpha_v)
        alpha_j = self.linear_alpha_j(alpha_j)
        beta_v = self.linear_beta_v(beta_v)
        beta_j = self.linear_beta_j(beta_j)
        '''

        # Concatenate all features
        # combined_features = torch.cat([tra_cdr3, epitope, trb_cdr3, alpha_v, alpha_j, beta_v, beta_j], dim=1)
        combined_features = torch.cat([tra_cdr3, epitope, trb_cdr3], dim=1)

        print(f"this is combined_features in forward(): \n{combined_features.shape}\n\n\n")
        # Apply Multi-Head Attention
        attn_output, _ = self.multihead_attn(combined_features, combined_features, combined_features)
        print(f"attn_output.shape: {attn_output.shape}")

        # proposal from ChatGPT
        pooled_attn_output = self.global_avg_pool(attn_output.transpose(1, 2)).squeeze(-1)  # Ensure it matches Linear layer input


        # Apply linear layers after attention
        post_attn_features = self.post_attn_linear(pooled_attn_output) # HERE ERROR...
        print(f"post_attn_features.shape: {post_attn_features.shape}")

        # Get final prediction
        logits = self.final_layer(post_attn_features)
        return logits
    

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        epitope_embedding = batch["epitope_embedding"]
        tra_cdr3_embedding = batch["tra_cdr3_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        label = batch["label"]

        # alpha_v = batch['alpha_v']
        # alpha_j = batch['alpha_j']
        # beta_v = batch['beta_v']
        # beta_j = batch['beta_j']
        
        # Process the data through the model
        # y_hat = self(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding, alpha_v, alpha_j, beta_v, beta_j)
        y_hat = self(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding)

        # Assuming 'y' is the target/output label in your batch
        y = label
        print(f"this is y in training_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss


    def test_step(self, batch, batch_idx):
        # Unpack the batch
        epitope_embedding = batch["epitope_embedding"]
        tra_cdr3_embedding = batch["tra_cdr3_embedding"]
        trb_cdr3_embedding = batch["trb_cdr3_embedding"]
        label = batch["label"]

        # alpha_v = batch['alpha_v']
        # alpha_j = batch['alpha_j']
        # beta_v = batch['beta_v']
        # beta_j = batch['beta_j']
        
        # Process the data through the model
        # y_hat = self(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding, alpha_v, alpha_j, beta_v, beta_j)
        y_hat = self(epitope_embedding, tra_cdr3_embedding, trb_cdr3_embedding)

        # Assuming 'y' is the target/output label in your batch
        y = label
        print(f"this is y in training_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        self.log('test_loss', loss)
        return loss
        


    '''
    # not yet implemented in train.py
    def validation_step(self, batch, batch_idx):
        # Implement validation logic here
        # Example:
        x, y = batch
        y_hat = self.forward(*x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    '''
    

    def configure_optimizers(self):
        # Configure optimizers and optionally schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    model = ModelV1()