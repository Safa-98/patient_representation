# Title to be added

## Requirements

### Dataset
MIMIC-III database analyzed in the study is available on PhysioNet repository. Here are some steps to prepare for the dataset:

* To request access to MIMIC-III, please follow https://mimic.physionet.org/gettingstarted/access/. Make sure to place ```.csv files``` under ``` data/mimic```.
* With access to MIMIC-III, to build the MIMIC-III dataset locally using Postgres, follow the instructions at https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres.
* Run SQL queries to generate necessary views, please follow https://github.com/onlyzdd/clinical-fusion/tree/master/query.



### Installing the Dependencies
Install Anaconda (or miniconda to save storage space).

Then, create a conda environement (for example stay-analogy) and install the dependencies, using the following commands:

```bash
$ conda create --name patient_repr python=3.9
$ conda activate patient_repr 
$ conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
$ conda install -y numpy scipy pandas scikit-learn
$ conda install -y tqdm gensim nltk
```

### Usage



  
 
  
  
## Files and Folders

- `data_processing.py`: contains clinical notes preprocessing steps to create final datatset
- `training.py`: contains the training and evaluate code of our classifiers
- **`models` folder**: contains all embedding models used to learn document-level representation
- **`patient_repr_aggregation` folder**: contains all codes used to experiment with different aggregation methods to learn patient-level representations


 
