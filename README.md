# SPSM
SPSM is a machine learning method to fit simple  submodels with missingness at test time in Python.

## Background 
SPSM lets users make quick predictions when missingness at the test time by having a shared coefficient and pattern-specific submodels. 

Here is an illustration that we included as a running example in our paper: 

![alt text](https://github.com/Healthy-AI/spsm/blob/main/SPSM_Example_.jpg)

## Reference
If you use SPSM in your research, we would appreciate a citation:
Stempfle, L., Panahi, A., & Johansson, F. D. (2023). Sharing Pattern Submodels for Prediction with Missing Values. Proceedings of the AAAI Conference on Artificial Intelligence, 37(8), 9882-9890. https://doi.org/10.1609/aaai.v37i8.26179

## Installation
´´´bash
create a new folder
git clone https://github.com/Healthy-AI/spsm.git

cd main_script 
python experiment.py -ds house_reg -es SPSM_ols -i none -pa alpha0 10.0 alphap 100.0 -sp 0.2 -s 0 -op True -m True -fr 1.0 #run example with housing dataset 
´´´
