---
title: 'Project documentation template'
disqus: hackmd
---

gNet: Global Early Exit for Pre-Trained Networks
===

gNet is a single network architecture that replaces multiple exit branches of the traditional early-exit architectures with a single global exit layer that supports anytime output from any layer of any pre-trained model. 

This folder contains all the code code needed to run the gNet architecture.

---

Folder Structure
===
```
EarlyExit
│   README.md  
└───CIFAR-10
│   │   Baseline-M.ipynb
│   │   Baseline-S.ipynb
|   |   Early Exit-M.ipynb
|   |   Early Exit-S.ipynb
└───CIFAR-1000
│   │   Baseline-M.ipynb
│   │   Baseline-S.ipynb
|   |   Early Exit-M.ipynb
|   |   Early Exit-S.ipynb
└───KWS
│   │   Baseline-S.ipynb
|   |   Early Exit-M.ipynb
|   |   Early Exit-S.ipynb
```
To Run
===
Train each baseline model and use the saved ```.h5``` checkpoint file for training early-exit models.




