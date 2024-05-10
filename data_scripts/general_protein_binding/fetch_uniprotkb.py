# %%
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import re
import time


# %%
general_protein_df = pd.read_csv("../data/UniProt/generalProteinBinding.tsv", sep="\t")


# %%
num_samples = 1000000
general_protein_df = general_protein_df.sample(n=num_samples)
general_protein_df

# %%
ORGANISM_HUMAN_ID = 9606

def fetch_uniprot_sequence(gene_symbol):
    url = "https://rest.uniprot.org/uniprotkb/search?"
    params = {
        "query": f'(gene_exact:"{gene_symbol}" AND organism_id:{ORGANISM_HUMAN_ID})',
        "fields": "sequence",
        "format": "fasta",
    }

    # Ensure we do not exceed the rate limit of 200 requests/min/user
    time.sleep(0.005)  # wait for 5 milliseconds before making a request

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        fasta_text = response.text
        entries = fasta_text.strip().split(">")
        entries = [entry for entry in entries if entry]
        if entries:
            # keep obnly 1. entry
            first_entry = entries[0]
            lines = first_entry.split("\n")
            # remove header
            first_sequence = ''.join(lines[1:])
            return first_sequence
        else:
            return "No sequences found"
    else:
        return f"Error: {response.status_code}"
    

'''
# for testing
gene_symbol = "MAP2K4"
sequence = fetch_uniprot_sequence(gene_symbol)
print(sequence)  
'''


# %%
proteins = []

# %%
to_path = "../data/GeneralProteinBinding/"
file_name = "general_proteins_23_04_24.tsv"
full_path = to_path + file_name


re_next_link = re.compile(r'<(.+)>; rel="next"')
retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
# Open the file at the beginning of your loop
with open(full_path, 'w') as file:  # Use 'w' to overwrite existing files or 'a' to append
    file.write("Protein 1 AA\tProtein 2 AA\n")  # Writing the header
    for index, row in general_protein_df.iterrows():
        seq1 = fetch_uniprot_sequence(row["OFFICIAL_SYMBOL_A"])
        seq2 = fetch_uniprot_sequence(row["OFFICIAL_SYMBOL_B"])
        # Writing each pair of sequences directly to the file
        file.write(f"{seq1}\t{seq2}\n")
