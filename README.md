SPSM

SPSM is a machine learning method to fit simple  submodels with missingness at test time in python.

Background 
SPSM let users make quick predictions when having missingness at the test time by having a shared coefficient and pattern specific submodels. 

Hier ist an illustration that we included as a running example in our paper: 

[figure]

Reference
If you use SPSM in your research, we would appreciate a citation to the following paper (LINK)!

Installation
git clone É

cd main_script 
python experiment.py -ds house_reg -es SPSM_ols -i none -pa alpha0 10.0 alphap 100.0 -sp 0.2 -s 0 -op True -m True -fr 1.0 #run example with housing dataset 


Requirements
SPSM requires [list of packages saved in the virtual environment]
