# botnet_active_learning
This repository contains code used for experiments in my BSc final thesis, “Multi-class Classification of Botnet Detection by Active Learning.”   

## Setup Instruction (For those who want to run the code)
### Environment
To get started, clone the repository:
``` 
git clone https://github.com/kei5uke/botnet-active-learning.git
```
Then, change your current directory and install the dependencies:  
```
cd active_learning
pip install -r requirements.txt
```
Next, install the MedBIoT and N-BaIoT datasets and store them in the ```/dataset``` directory   
The file structure is shown in the directory, so be sure to install the datasets accordingly     
You can find the datasets here:   
- [MedBIoT Dataset](https://cs.taltech.ee/research/data/medbiot/)  
- [N-BaIoT Dataset](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot)  

### Dataset Pickels
Only a small portion of the datasets is used for the experiments  
To generate dataset pickles, run ```python3 Make_df_MedBIoT.py``` and ```python3 Make_df_N-BaIoT.py```  

### Experiment
Now you are ready to run any experimental code in ```/active_learning``` directory.
