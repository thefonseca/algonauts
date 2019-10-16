# PredNet - Algonauts
Deep predictive coding networks are neuroscience-inspired unsupervised 
learning models that learn to predict future sensory states. We build upon the 
[PredNet](https://coxlab.github.io/prednet/) implementation by 
Lotter, Kreiman, and Cox (2016) to investigate if predictive coding 
representations are useful to predict brain activity in the visual cortex. 
We use representational similarity analysis (RSA) to compare PredNet 
representations to functional magnetic resonance imaging (fMRI) and 
magnetoencephalography (MEG) data from the Algonauts Project (Cichy et al., 2019).  

In contrast to previous findings in the literature (Khaligh-Razavi & Kriegeskorte, 2014), 
we report empirical data suggesting that unsupervised models trained to predict frames 
of videos may outperform supervised image classification baselines. 

## Code

This repository contains supporting code for PredNet training, fine-tuning, 
feature extraction, and evaluation. We also use the [Algonauts development kit](http://algonauts.csail.mit.edu/challenge.html), 
which is not distributed here. Experiment workflow is as follows:
 
* [PredNet training](./prednet_train.ipynb) on videos
* [Feature extraction](./prednet_features.ipynb)
* [Feature extraction for CORnet](./cornet_features.ipynb) (Kubilius et al. 2018)
* [Evaluation](./prednet_evaluation.ipynb) using representational dissimilarity analysis (RSA) 
