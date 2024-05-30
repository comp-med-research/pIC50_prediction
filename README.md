# pIC50_prediction
Developing ML models to predict pIC50 levels 


During the course of a drug discovery program, a critical task is the ability to “screen” a
library of compounds in order to find molecules that can bind to and potentially inhibit
the activity of a target protein (we call such readout “potency”). Due to the prohibitive
cost of large scale experimental screening, virtual in silico screening serves as an initial
step. This approach significantly reduces costs while facilitating the evaluation and
prioritization of an extensive range of small molecules.

A variety of methods is available for virtual screening, including ligand-based machine
learning models that rely on the molecular structure as input to predict their activities.
This notebook includes an exploration of a dataset of 4.6k compounds that have undergone 
experimental testing against the Epidermal Growth Factor Receptor (EGFR) kinase, a target
associated with various cancers, as well as a prediction of the potency value (pIC50), using 
an pretrained foundation model called SELFormer to produce embeddings.
