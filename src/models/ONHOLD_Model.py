import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel, pipeline
import re

def createEmbedding(aa_sequence, device): 
    # TODO: consider to fine-tune the Prot-T5-XL model while training
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    if device==torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32) # only cast to full-precision if no GPU is available
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, legacy=True)

    # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    processed_epitope = " ".join(list(re.sub(r"[UZOB]", "X", aa_sequence)))

class DummyModel(nn.Module): 
    def __init__(self): 
        super(DummyModel, self).__init__()
