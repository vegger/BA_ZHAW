## Authors
- [@vegger](https://www.github.com/vegger)
- [@cyrilgabriele](https://www.github.com/cyrilgabriele)

## About this project
This project is part of the authors' Bachelor's thesis, which is about predicting TCR-Epitope binding affinity for advanced immunotherapy. The aim of this thesis is to develop machine learning models capable of predicting the binding between T cell receptors (TCRs) and peptide-major histocompatibility complexes (pMHCs), thereby advancing personalized immunotherapy. By leveraging deep learning techniques, the project addresses the challenges posed by the highly individualistic nature of the human immune system and the lack of diverse, comprehensive datasets.

### Data Sources
The primary sources of data include [VDJdb](https://vdjdb.cdr3.net/), [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/), and [IEDB](https://www.iedb.org/), which provide sequences and true postitive binding data for TCRs and pMHCs.

### Data Processing
The data is standardized, harmonized, and split into training, validation, and test sets. Negative samples are synthetically generated to ensure a balanced dataset. The [Data Pipeline](#run-data-pipeline) section explains how you can run the data pipeline locally.

### Model Architectures
Various deep learning architectures are explored, including attention-based models. The [Model Training](#train-a-model) section explains how the training works in this project.

### Repository Structure
`data/`: This will be used to store data locally\
`data_scripts/`: Contains all scripts related to data acquisition, preprocessing and analyzing\
`models/`: Includes different model architectures and training scripts\

## Prerequisites
The following requirements must be met in order to work with this project.

### Hardware
Make sure you have a proper GPU and CUDA (version 12.1) installed. Other CUDA versions may work too but the [PyTorch](https://pytorch.org/get-started/locally/) installation can lead to problems. The NVIDIA GeForce GTX 1650 for example is not sufficient and leads to 'CUDA out of memory' issues. 

### Weights and Biases account
The [Weights and Biases](https://wandb.ai/site) account is used as MLOps Plattform to store datasets and do some model tuning.

### Environment Variables
To run this project, you will need to add the following environment variable to your .env file.\
`MAIN_PROJECT_NAME`\
This environment variable should reflect how your project is named in your Weights & Biases account.

### Conda Environment
We recommend to have Anaconda installed which provides package, dependency, and environment management for any language. To import the conda environment, execute the following command in the root folder of this project and activate it.
The name of the environment should be preserved and is called BA_ZHAW.
```bash
conda env create -n BA_ZHAW --file ENV.yml
conda activate BA_ZHAW
```
Install the necessary pip packages.
```bash
pip install tidytcells
pip install peptides
```
As the pytorch installation isn't cross-compatible with every device, we suggest to reinstall it properly. First uninstall it.
```bash
conda uninstall pytorch -y
```
Now pytorch can be reinstalled. Therfore check the [Pytorch Documentation](https://pytorch.org/get-started/locally/)

#### Conda Issues
Sometimes the replication of conda environments does not work as good as we may wish. In this create a new environment with python version 3.12 or higher.
The following list should cover all the needed packages without guarantee of completeness. It will certainly prevent the vast majority of ModuleNotFound errors.
First install [Pytorch Documentation](https://pytorch.org/get-started/locally/) and then:
```
conda install numpy
pip install python-dotenv
pip install nbformat
pip install tidytcells
conda install pandas
pip install peptides
conda install wandb --channel conda-forge
conda install conda-forge::pytorch-lightning
conda install matplotlib
conda install -c conda-forge scikit-learn
conda install conda-forge::transformers
```
In some cases pytorch needs to have [sentencepiece](https://pypi.org/project/sentencepiece/) installed. When you work with cuda version 12.2 and have PyTorch installation for cuda version 12.1 installed, you will need it for sure. 
```
pip install sentencepiece
```
## Run Locally
- Clone the project
```bash
  git clone https://github.com/vegger/BA_ZHAW.git
```
- Create conda environment as explained above and use it from now on
- Open the project in the IDE of your choice
- Ensure the project is set as the root directory in your IDE. Otherwise, you may encounter path errors when running commands like %run path/to/other_notebook.ipynb.

### Run Data Pipeline
- place the [plain_data](https://www.dropbox.com/scl/fo/ucke53zlkj9sau6qlg63q/ANL6-gCocJ5sj_zs0T59CVI?rlkey=ogbq0p0zedpef29fif1ihfs3u&st=n2e2x3d5&dl=0) folder in the data folder, where the README_PLAIN_DATA.md is located.
- In order to execute the data pipeline, which harmonizes and splits data, then creates embeddings and PhysicoChemical properties, do the following:
  - Open data_pipeline.ipynb in the root folder
  - set the variable precision to `precision="allele"` or `precision="gene"`
  - in some environments like [lightning.ai](https://lightning.ai/), the `pipeline_data = './data'` needs to have an absolute path instead of the relative.
  - Run the notebook with the newly created conda environment
  - The output is placed in the `./data` folder
  - The final split for beta paired datasets can be found under `./data/splitted_data/{precision}/ `
  - Run the notebook again with different precision to create all datasets

### Train a Model
- There are four scripts to do training. Each can be run with gene or allele precision (make sure datapipeline has been run with the corresponding precision).
  - `./models/beta_physico/train_beta_physico.py`
  - `./models/beta_vanilla/train_beta_vanilla.py`
  - `./models/physico/train_physico.py`
  - `./models/vanilla/train_vanilla.py`
- Open the train skript of your choice and head to the top of the main function.
  - set value for the variable `precision`
  - If you had to change to an absolute path in the data pipeline:
    - change `embed_base_dir` to an absolute path
    - change `physico_base_dir` to an absolute path if you train either `train_beta_physico.py` or `train_physico.py`
  - If you want to do hyperparameter tuning with Weights & Biases sweeps
    - change `hyperparameter_tuning_with_WnB` to True
  - Otherwise set the specific hyperparameter values in the train script:
  
    ```
    # ! here random hyperparameter values set !
    hyperparameters["optimizer"] = "adam"
    hyperparameters["learning_rate"] = 5e-3
    hyperparameters["weight_decay"] = 0.075
    hyperparameters["dropout_attention"] = 0.3
    hyperparameters["dropout_linear"] = 0.45
    ```
    
  - After training one can see the checkpoint file (`.ckpt`) in the directory `checkpoints` in a directory named like the Weights & Biases run. The checkoint is saved at the point where the AP_Val metric was at its highest. Furthermore, the file with the `.pth` extension is the final model. These files are in the same directory as the training script.

## Additional Data
Prebuilt Embeddings, Models, ModelRuns and Physicochemical Properties are shared over [sharepoint](https://zhaw-my.sharepoint.com/:u:/g/personal/eggerval_students_zhaw_ch/EaZpwuhuUn9DpY6PcXrmrgEB5K-Qw5Git-W7o914mMRa_w?e=dQxMjw). Feel free to download.<br>
The provided models are checkpoints evaluated at the AP_Val maximum point. These need to be trained further to get the same outputs as in the work. This has been decided so that they can be used as a starting point for task-specific training (TPP).

## Disclaimer
- The data pipeline and the model trainings were executed in the [lightning.ai](https://lightning.ai/) environment. Sometimes there were Cuda memory errors, then the VM had to be restarted for it to work again.
- In Windows environment it was observed that the creation of the PhysicoChemical properties can lead to problems, in this case Assertion Error is thrown, which says that a scalar has the length 88 instead of 101. The authors are thrilled to know why.
- **Attention** if one uses a legacy checkpoint of the Paired Vanilla model (vanilla_model.py) provided in the shared files. The checkpoints contain a unnecessary building block, namely "multihead_attn_physico" which must **NOT** be used. Filter therefore the state_dict, as shown after if one would use the checkpoints to reproduce the results:
  
  `filtered_state_dict = {k: v for k, v in state_dict.items() if "multihead_attn_physico" not in k}`
