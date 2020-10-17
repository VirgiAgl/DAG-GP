# DAG-GP
Code for the paper Multi-task Causal Learning with Gaussian Processes (https://arxiv.org/pdf/2009.12821.pdf)

# Usage
1) The code to run the DAG-GP model is in `runCTF.py`. The file `runGPregression.py` can be used to get results for the single-task GP models which are used as benchmarks in the paper.

2) The code for AL with the DAG-GP model is in `runAL_CTF.py`. The file `runAL_reg.py` can be used to get results for AL with single-task GP models which are used as benchmarks in the paper.

3) The code for running CBO with the DAG-GP model is in `runCBO.py`. By modifyng the value of the variable `model_type` one can change the surrogate GP model used in the algorith. When `model_type==1` the DAG-GP model is used. When `model_type==0` a single-task GP model is used.

For all experiments it is possible to change the causal graph that by changing the value of the variable `experiment`. In addition, it is possible to modify the GP prior used by changing the value of the variable `causal_prior`. The results are saved in the folder `Data/` when running the experiments. The folder `Data/` contains the observational and intervational data used to produce the results in the paper. 

# Contacts
Feel free to contact the first author of the paper ([Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/)) for questions 

