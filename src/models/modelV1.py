import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC


class ModelV1(pl.LightningModule):
    def __init__(self, batch_size, embed_dim, max_seq_length):
        super(ModelV1, self).__init__()
        # for evaluation of the model
        self.auroc = AUROC(task="binary")

        self.batch_size = batch_size

        # for asserts

        # CNNs
        # must be: (N, C_in, L) where: 
        # N: Batch Size 
        # C_in: Emb. dim. 
        # L = max AA length 
        self.conv_shape = [batch_size, embed_dim, max_seq_length]

        # Transformers
        # must be: (L, N, E) where: 
        # L: max AA length 
        # N: Batch Size 
        # E: Emb. dim. 
        self.transformer_shape = [max_seq_length, batch_size, embed_dim]


        # Define Conv1D layers for αCDR3, Epitope, βCDR3
        self.embed_dim = embed_dim
        self.out_channels = 1024//3
        self.kernel_size = 3
        self.drop_out_p = 0.3
        
        # must embed_dim must be dividable by num_heads!
        self.num_heads = 33
        self.in_features1 = 1023
        self.out_features1 = 1024
        self.in_features2 = 1024
        self.out_features2 = 1

        self.conv1d_epitope = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
        self.conv1d_alpha = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")
        self.conv1d_beta = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.out_channels, kernel_size=self.kernel_size, padding="same")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.drop_out_p )

        '''
        # Define linear layers for aV, aJ, bV, bJ
        self.linear_alpha_v = nn.Linear(in_features=..., out_features=...)
        self.linear_alpha_j = nn.Linear(in_features=..., out_features=...)
        self.linear_beta_v = nn.Linear(in_features=..., out_features=...)
        self.linear_beta_j = nn.Linear(in_features=..., out_features=...)
        '''

        self.multihead_attn = nn.MultiheadAttention(embed_dim=1023, num_heads=self.num_heads)
        # TODO: does not work properly...
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Reduces each feature to a single number
        self.post_attn_linear = nn.Linear(in_features=26598, out_features=self.out_features1)
        self.final_layer = nn.Linear(in_features=self.in_features2, out_features=self.out_features2)
        

    # def forward(self, tra_cdr3, epitope, trb_cdr3, alpha_v, alpha_j, beta_v, beta_j):
    def forward(self, epitope, tra_cdr3, trb_cdr3):        
        print(f"epiotpe.shape: {epitope.shape}")
        print(f"tra_cdr3.shape: {tra_cdr3.shape}")
        print(f"trb_cdr3.shape: {trb_cdr3.shape}")

        print("after permutation:")

        print(f"epiotpe.shape: {epitope.permute(0, 2, 1).shape}")
        print(f"tra_cdr3.shape: {tra_cdr3.permute(0, 2, 1).shape}")
        print(f"trb_cdr3.shape: {trb_cdr3.permute(0, 2, 1).shape}")

        epitope_permuted = epitope.permute(0, 2, 1)
        tra_cdr3_permuted = tra_cdr3.permute(0, 2, 1)
        trb_cdr3_permuted = trb_cdr3.permute(0, 2, 1)

        '''
        as batch size is different for validation and test not testable like this...
        assert epitope_permuted.shape == self.conv_shape, "Epitope shape wrong for Conv!"
        assert tra_cdr3_permuted.shape == self.conv_shape, "TRA_CDR3 shape wrong for Conv!"
        assert trb_cdr3_permuted.shape == self.conv_shape, "TRB_CDR3 shape wrong for Conv!"
        '''
        
        # Apply conv1d + ReLU + Dropout for Epitope, αCDR3, βCDR3
        epitope = self.dropout(self.relu(self.conv1d_epitope(epitope_permuted)))
        tra_cdr3 = self.dropout(self.relu(self.conv1d_alpha(tra_cdr3_permuted)))
        trb_cdr3 = self.dropout(self.relu(self.conv1d_beta(trb_cdr3_permuted)))

        print("After: Apply conv1d + ReLU + Dropout for Epitope, aCDR3, bCDR3")
        print(f"epiotpe.shape: {epitope.shape}")
        print(f"tra_cdr3.shape: {tra_cdr3.shape}")
        print(f"trb_cdr3.shape: {trb_cdr3.shape}")
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

        print(f"combined_features.shape: {combined_features.shape}\n\n\n")

        # Apply Multi-Head Attention
        attn_input = combined_features.permute(2, 0 , 1)
        # assert attn_input.shape == self.transformer_shape
        print(f"attn_input.shape: {attn_input.shape}")
        attn_output, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        print(f"attn_output.shape: {attn_output.shape}\n\n\n")

        global_pool_input = attn_output.permute(1, 2, 0)
        print(f"global_pool_input.shape: {global_pool_input.shape}")

        print(f"self.global_avg_pool(global_pool_input): {self.global_avg_pool(global_pool_input).shape}")
        # TODO: this does not work somehow!! i got the same dimension in as out...
        # pooled_attn_output = self.global_avg_pool(global_pool_input).squeeze(-1)  # Ensure it matches Linear layer input
        pooled_attn_output = global_pool_input.flatten(1)
        # attn_output = attn_output.transpose(0, 1).flatten(1)
        print(f"pooled_attn_output: {pooled_attn_output.shape}")

        # Apply linear layers after attention
        post_attn_features = self.post_attn_linear(pooled_attn_output) 
        print(f"post_attn_features.shape: {post_attn_features.shape}")

        # Get final prediction
        logits = self.final_layer(post_attn_features)
        print(f"logits: {logits}")
        return logits
    

    def training_step(self, batch, batch_idx):
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

        y = label
        # print(f"this is y in training_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss


    def test_step(self, batch, batch_idx):
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
        print(f"this is y_hat in test_step: {y_hat}\n\n\n")

        y = label
        # print(f"this is y in test_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        self.log('test_loss', loss)

        predictions = torch.sigmoid(y_hat)
        print(f"predictions in test: {predictions}")
        # self.log('test_auroc', self.auroc(probas, label.int()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ROCAUC Test", self.auroc(predictions, y.unsqueeze(1)), on_epoch=True, prog_bar=True)
        return loss
        


    
    def validation_step(self, batch, batch_idx):
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
        # print(f"this is y_hat in test_step: {y_hat}\n\n\n")
        # Assuming 'y' is the target/output label in your batch
        y = label
        # print(f"this is y in test_step: {y}\n\n\n")
        loss = F.binary_cross_entropy_with_logits(y_hat, y.unsqueeze(1))
        self.log('validation_loss', loss)
        loss = F.binary_cross_entropy_with_logits(y_hat, label.unsqueeze(1))
        
        predictions = torch.sigmoid(y_hat)
        # self.log('val_auroc', self.auroc(probas, label.int()), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("ROCAUC Validation", self.auroc(predictions, y.unsqueeze(1)), on_step=False, on_epoch=True, prog_bar=True,)
        return loss
    
    

    def configure_optimizers(self):
        # Configure optimizers and optionally schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    model = ModelV1()