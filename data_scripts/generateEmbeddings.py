import pandas as pd
import numpy as np
import gc
import argparse
import re
import torch 
import os
from transformers import pipeline, T5Tokenizer, T5EncoderModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))


#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision)
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
gc.collect()
if device==torch.device("cpu"):
    print("Casting model to full precision for running on CPU ...")
model.to(torch.float32) # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True)

def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t")

    return df

def process_batch(processed_seqs):
    # Extract just the processed sequences for tokenization
    sequences = [seq[1] for seq in processed_seqs]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest", return_tensors="pt")
    input_ids = ids['input_ids'].to(device)
    attention_mask = ids['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    last_hidden_states = outputs.last_hidden_state
    
    embeddings = {}
    for i, (original_seq, _) in enumerate(processed_seqs):
        seq_len = len(original_seq)
        valid_embeddings = last_hidden_states[i,:seq_len]
        embeddings[original_seq] = valid_embeddings.cpu().numpy()

    return embeddings
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for protein sequences.")

    parser.add_argument('chain', type=str, help="The value is paired or beta")
    parser.add_argument('input_file_name', type=str, help="The file name of the dataset.")
    parser.add_argument('output_file_name', type=str, help="The name of the output file (TRB_beta_embeddings.npz)")
    parser.add_argument('column_name', type=str, help="The column name containing the sequences.")

    args = parser.parse_args()

    df = load_data(args.input_file_name)
    output_file_name = args.output_file_name
    sequences = set(df[args.column_name].to_list())
    processed_sequences = [(sequence, " ".join(list(re.sub(r"[UZOB]", "X", sequence)))) for sequence in sequences]

    batch_size = 128
    sequence_to_embedding = {}

    # Batch processing with a dictionary, using original sequences as keys
    for i in range(0, len(processed_sequences), batch_size):
        batch_sequences = processed_sequences[i:i+batch_size]
        batch_embeddings = process_batch(batch_sequences)
        sequence_to_embedding.update(batch_embeddings)

    if not os.path.exists(os.path.dirname(output_file_name)):
      os.makedirs(os.path.dirname(output_file_name))
    np.savez(output_file_name, **sequence_to_embedding)
