import peptides
import pandas as pd
import numpy as np
import torch
import gc
import multiprocessing as mp

def process_batch(sequences):
    result = {}
    for sequence in sequences:
        sequence_Peptide_obj = peptides.Peptide(sequence)
        properties = {}
        properties["QSAR"] = sequence_Peptide_obj.descriptors()
        properties["aliphatic_index"] = sequence_Peptide_obj.aliphatic_index()
        table = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
        properties["autocorrelation"] = sequence_Peptide_obj.auto_correlation(table=table)
        properties["autocovariance"] = sequence_Peptide_obj.auto_covariance(table=table)
        properties["boman_index"] = sequence_Peptide_obj.boman()
        properties["lehninger_charge"] = sequence_Peptide_obj.charge(pKscale="Lehninger")
        alpha = 100  # if angle = 100° -> hydrophobic moment alpha, according to the doc
        properties["hydrophobic_moment_alpha"] = sequence_Peptide_obj.hydrophobic_moment(angle=alpha)
        beta = 160  # if angle = 160° -> hydrophobic moment beta, according to the doc
        properties["hydrophobic_moment_beta"] = sequence_Peptide_obj.hydrophobic_moment(angle=beta)
        properties["hydrophobicity"] = sequence_Peptide_obj.hydrophobicity(scale="KyteDoolittle")
        properties["instability_index"] = sequence_Peptide_obj.instability_index()
        properties["isoelectric_point"] = sequence_Peptide_obj.isoelectric_point(pKscale="EMBOSS")
        properties["mass_shift"] = sequence_Peptide_obj.mass_shift(aa_shift="silac_13c")
        properties["molecular_weight"] = sequence_Peptide_obj.molecular_weight(average="expasy")
        properties["mass_charge_ratio"] = sequence_Peptide_obj.mz()

        all_feature_values = []
        for value in properties.values():
            if isinstance(value, dict):
                all_feature_values.extend(value.values())
            else:
                all_feature_values.append(value)

        feature_array = np.array(all_feature_values, dtype=np.float32)
        result[sequence] = feature_array  # store as numpy array

    return result

def parallel_compute_properties(df, column_name):
    sequences = list(set(df[column_name].to_list()))
    batch_size = 128
    pool = mp.Pool(mp.cpu_count())

    results = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        result = pool.apply_async(process_batch, [batch])
        results.append(result)

    pool.close()
    pool.join()

    sequence_properties = {}
    for result in results:
        sequence_properties.update(result.get())

    for sequence in sequence_properties:
        sequence_properties[sequence] = torch.tensor(sequence_properties[sequence])

    return sequence_properties

def main():
    # MOST RECENT
    dataset = "train"
    precision = "allele"
    df = pd.read_csv(f"/teamspace/studios/this_studio/BA_ZHAW/models/physico/WnB_Experiments_Datasets/paired_{precision}/{precision}/{dataset}.tsv", sep="\t")
    
    
    epitope_properties = parallel_compute_properties(df, "Epitope")
    np.savez(f"./{dataset}_paired_epitope_{precision}_physico.npz", **epitope_properties)
    
    gc.collect()
    
    tra_properties = parallel_compute_properties(df, "TRA_CDR3")
    np.savez(f"./{dataset}_paired_TRA_{precision}_physico.npz", **tra_properties)  
    
    gc.collect()
    
    trb_properties = parallel_compute_properties(df, "TRB_CDR3")
    np.savez(f"./{dataset}_paired_TRB_{precision}_physico.npz", **trb_properties)    
    
    gc.collect()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
