# botnet_active_learning
This repository contains code used for experiments in my BSc final thesis, “Multi-class Classification of Botnet Detection by Active Learning.”   

## Thesis in Nutshell 
Labeling malware samples and network traffic is an expensive process in the cybersecurity industry. This active learning framework allows for the efficient development of effective machine learning models using a limited dataset. This thesis compares well-known query strategies to identify which strategy and parameters yield the best results while minimizing the required data samples.

### Figure 1. Cycle of Active Learning
![AL_cycle](https://github.com/kei5uke/botnet-active-learning/assets/33390452/01de7ca5-090d-4658-ab70-89e8f5a0ea36) 

### Figure 2. Uncertainty Sampling VS Query by Committee VS Random Sampling
![US_QbC](https://github.com/kei5uke/botnet-active-learning/assets/33390452/257e1dce-2503-4f45-bda5-99c383921190)    

### Figure 3. Ranked Batch-mode Sampling VS Random Sampling
![Ranked](https://github.com/kei5uke/botnet-active-learning/assets/33390452/a51daa0d-c6c0-4fa3-9cf8-ba0fb5ef6cfe)    

### Conclusion
- Margin Sampling is the optimal strategy in terms of stability and convergence speed.
- If multiple instances are required in each iteration, Ranked Batch-mode Sampling with a small unlabeled pool may perform well.

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
Change common variables in ```global_variable.py``` and shared variables in each file  
Now you are ready to run any experimental code in ```/active_learning``` directory  

