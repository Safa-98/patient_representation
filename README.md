# Deep learning-based patient note-identification using clinical documents
This repository contains the code and resources for _Deep learning-based patient note-identification using clinical documents_. The project aims to address the challenge of patient-note identification, where we try to accurately associate a single clinical note with its corresponding patient. This task highlights the importance of learning robust patient-level representations. We propose different embedding models and experiment with various aggregation techniques to produce robust patient-level representations. Our models are implemented using PyTorch. 


## Repository Structure


```bash
project-root/
│
├── data_processing.py          # Script to obtain and preprocess the dataset
│
├── data/
│   ├── raw/                    # Original dataset files and CSVs
│   ├── processed/              # Files with embeddings
│
├── models/                     # Embedding models to learn representations
│
├── patient_repr_aggregation/   # Codes for learning patient embeddings
│ 
├── training.py                 # Script to train and evaluate the classifier
│ 
├── results/                    # Results from model evaluations
│
├── .gitattributes              # Configures repository attributes like line endings
├── .gitignore                  # Specifies files and directories to ignore
└── README.md                   # Project description and instructions

```

## Requirements

### Dataset
MIMIC-III database analyzed in the study is available on PhysioNet repository. Here are some steps to prepare for the dataset:

* To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/. Make sure to place ```.csv files``` under ``` data/mimic```.
* With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres.
* Run SQL queries to generate necessary views, please follow https://github.com/onlyzdd/clinical-fusion/tree/master/query.



### Installing the Dependencies
Install Anaconda (or miniconda to save storage space).

Then, create a conda environement (for example patient_repr) and install the dependencies, using the following commands:

```bash
$ conda create --name patient_repr python=3.9
$ conda activate patient_repr 
$ conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
$ conda install -y numpy scipy pandas scikit-learn
$ conda install -y tqdm gensim nltk
```

### Usage

**Data Processing:**

*Run the data processing script to prepare the dataset.*
```bash

python data_processing.py
```
**Model Training:**

*Train and evaluate the classifier using the processed data.*

```bash
python training.py
```

**Results:**

*The results of the model evaluation will be saved in the results/ directory.*

  
## Files and Folders

- `data_processing.py`: contains clinical notes preprocessing steps to create final datatset
- `training.py`: contains the training and evaluate code of our classifiers
- **`data` folder**: contains raw and processed datasets
- **`models` folder**: contains all embedding models used to learn document-level representation
- **`patient_repr_aggregation` folder**: contains all codes used to experiment with different aggregation methods to learn patient-level representations
- **`results` folder**: contains the results obtain by our classifier

 
