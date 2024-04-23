import pandas as pd
import numpy as np
import gc
import argparse
import re
import torch 
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


def load_data(file_name): 
    base_path = "../data/customDatasets/Embeddings/"
    df = pd.read_csv(base_path+file_name, sep="\t")

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
    
    # Now, return embeddings mapped to the original sequence
    embeddings = {}
    for i, (original_seq, _) in enumerate(processed_seqs):
        seq_len = len(original_seq)
        valid_embeddings = last_hidden_states[i,:seq_len]
        per_protein_embedding = valid_embeddings       
        embedding = per_protein_embedding.cpu().numpy()
        embeddings[original_seq] = embedding  # Use original sequence as key

    return embeddings
    

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generate embeddings for protein sequences.")

    # Add the arguments
    parser.add_argument('file_name', type=str, help="The file name of the dataset.")
    parser.add_argument('column_name', type=str, help="The column name containing the sequences.")
    parser.add_argument('--prefix', type=str, default="embeddings_", help="Prefix for the output file.")

    # Execute the parse_args() method
    args = parser.parse_args()

    df = load_data(args.file_name)
    sequences = set(df[args.column_name].to_list())


    processed_sequences = [(sequence, " ".join(list(re.sub(r"[UZOB]", "X", sequence)))) for sequence in sequences]

    batch_size = 128
    sequence_to_embedding = {}

    # Batch processing with a dictionary, using original sequences as keys
    for i in range(0, len(processed_sequences), batch_size):
        batch_sequences = processed_sequences[i:i+batch_size]
        batch_embeddings = process_batch(batch_sequences)
        sequence_to_embedding.update(batch_embeddings)

    to_path = "../data/Embeddings/"
    file_name = args.prefix + "embeddings.npz"
    np.savez(to_path+file_name, **sequence_to_embedding)


