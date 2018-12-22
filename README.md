# Contrastive-Predictive-Coding

This is a PyTorch implementation of [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)

Use main.py to train CPC features using CNN encoder. You can change various parameters using command-line arguments as given in the code.

Use main2.py to train CPC features using Resnet encoder. You can change various parameters using command-line arguments as given in the code.

Use classifier.py and classifier_with_resnet.py for the classification after CPC training.

Use make_clusters.py to get features and targets as pickle files.

kmeans.py is used to plot the features.

