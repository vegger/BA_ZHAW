# %%
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, T5Tokenizer, T5EncoderModel
import torch
import re
import numpy as np
import sys

# %%
read_path = "../../data/customDatasets/Stitchr_beta_concatenated.tsv"
stitchr_beta_df = pd.read_csv(read_path, sep="\t")

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

# %%

#@title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision)
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
if device==torch.device("cpu"):
  print("Casting model to full precision for running on CPU ...")
  model.to(torch.float32) # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True)


# %%
epitopes = set(stitchr_beta_df["Epitope"].to_list())

# %%
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
processed_epitopes = [(sequence, " ".join(list(re.sub(r"[UZOB]", "X", sequence)))) for sequence in epitopes]
# processed_epitopes

# %%
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
        per_protein_embedding = valid_embeddings.mean(dim=0)        
        embedding = per_protein_embedding.cpu().numpy()
        embeddings[original_seq] = embedding  # Use original sequence as key

    return embeddings

# %%

to_path = "../../data/customDatasets/negative_samples/temp/"
file_name = "negative_samples_Stitchr_beta_embeddings_dict.npz"


# %%

batch_size = 128
sequence_to_embedding = {}

# Batch processing with a dictionary, using original sequences as keys
for i in range(0, len(processed_epitopes), batch_size):
    batch_sequences = processed_epitopes[i:i+batch_size]
    batch_embeddings = process_batch(batch_sequences)
    sequence_to_embedding.update(batch_embeddings)


# %%
np.savez(to_path+file_name, **sequence_to_embedding)

# %%
epitope_to_embedding = np.load(to_path+file_name)

# %%
max_index = len(stitchr_beta_df) - 1 
negative_epitopes_cosine = []

# %%
def cosine_similarity(embedding1, embedding2): 
    cosine = np.dot(embedding1,embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
    return cosine

# %%
def is_valid_negative(cosine_similarity, current_epitope, random_epitope): 
    is_valid = False
    cosine_min = -1
    cosine_max = 0.75

    if (cosine_similarity >= cosine_min \
        and cosine_similarity <= cosine_max) \
        and (current_epitope != random_epitope): 
        is_valid = True 

    return is_valid


# %%
sys_max_depth = sys.getrecursionlimit()
max_attempts_by_system = sys_max_depth - 1

# %%
np.random.seed(42) 

# %%
def search_negative_epitope_embedding(df, index, current_epitope, max_attempts=max_attempts_by_system): 
    current_epitope = df["Epitope"][index]
    current_embedding = epitope_to_embedding[current_epitope]
    attempt = 0
    
    while attempt < max_attempts:
        random_epitope_index = np.random.randint(0, len(df))
        random_epitope = df["Epitope"][random_epitope_index]
        random_mhc_a = df["MHC A"][random_epitope_index]
        random_mhc_b = df["MHC B"][random_epitope_index]
        
        if random_epitope_index == index:
            attempt += 1
            continue  # Skip the rest of the loop and try again
        
        random_epitope_embedding = epitope_to_embedding[random_epitope]
        cosine = cosine_similarity(current_embedding, random_epitope_embedding)
        
        if is_valid_negative(cosine, current_epitope, random_epitope) or attempt == max_attempts - 1:
            return (random_epitope, random_mhc_a, random_mhc_b)  # Return the found valid or last attempt epitope
        
        attempt += 1
    
    # This point should theoretically never be reached because of the check in the loop,
    # but it's a fallback to return a random different epitope if for some reason it does.
    while True:
        random_epitope_index = np.random.randint(0, len(df))
        if random_epitope_index != index:
            return df["Epitope"][random_epitope_index]


# %%
for i, epitope in enumerate(stitchr_beta_df["Epitope"]):
    negative_epitope = search_negative_epitope_embedding(stitchr_beta_df, i, epitope)
    negative_epitopes_cosine.append(negative_epitope)

# %%
print(len(negative_epitopes_cosine)) # should be: 176'852

# %%
epitopes = []
mhc_a = []
mhc_b = []

# %%
for row_infos in negative_epitopes_cosine:
    epitopes.append(row_infos[0]) 
    mhc_a.append(row_infos[1])
    mhc_b.append(row_infos[2])

# %%
negative_epitopes_cosine_dict = {"Negative Epitope": epitopes, "MHC A": mhc_a, "MHC B": mhc_b}
negative_epitopes_cosine_df = pd.DataFrame(negative_epitopes_cosine_dict)
# print(negative_epitopes_cosine_df.to_string())
# print(negative_epitopes_cosine_df.to_string())

# %%
negative_epitopes_cosine_df["Negative Epitope"][0]

# %%
stitchr_beta_negative_epitope_df = stitchr_beta_df.drop(["MHC A", "MHC B"], axis=1).copy(deep=True)
stitchr_beta_negative_epitope_df["Epitope"] = epitopes
stitchr_beta_negative_epitope_df["MHC A"] = mhc_a
stitchr_beta_negative_epitope_df["MHC B"] = mhc_b
stitchr_beta_negative_epitope_df["Binding"] = 0

# %%
stitchr_beta_with_negative_df = pd.concat([stitchr_beta_df.copy(deep=True), stitchr_beta_negative_epitope_df], axis=0)

# %%
columns_to_ignore_for_duplicates = stitchr_beta_with_negative_df.columns.difference(["TCR_name", "Binding"])
stitchr_beta_with_negative_df.drop_duplicates(inplace=True, subset=columns_to_ignore_for_duplicates, keep="first")
stitchr_beta_with_negative_df["TCR_name"] = range(1, len(stitchr_beta_with_negative_df)+1)
stitchr_beta_with_negative_df.reset_index(drop=True, inplace=True)
stitchr_beta_with_negative_df

# %%
to_path = "../../data/customDatasets/negative_samples/"
file_name = "Stitchr_beta_concatenated_with_negative.tsv"

# %%
stitchr_beta_with_negative_df.to_csv(to_path+file_name, sep="\t", index=False)


