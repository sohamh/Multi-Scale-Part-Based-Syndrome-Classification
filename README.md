# Multi-Scale-Part-Based-Syndrome-Classification
This repository contains the source code, trained models and all data needed  to obtain classification results related to the article entitled "Multi-Scale Part-Based Syndrome Classification of 3D Facial Images".

Geometric_Encoder folder contains the implementation of the geometric encoder in PyTorch. To replicate the model in the paper, model parameters can be chosen as explained in the paper. All parameters can be changes in from encoder/arguments.py.

Embeddings folder contains all embeddings obtained from GE and PCA, based on the data that is described in the paper. Low dimensional embeddings are reproducible for the portion of the data that is publicly available (The FaceBase repository (www.facebase.org)). To pre-process the data, MeshMonk software is required as explained in the pre-processing section of the paper.

Classification folder contains code for producing the classification results. And Results folder contains the results obtained based on the embeddings.

Reference:
S. S. Mahdi et al., "Multi-Scale Part-Based Syndrome Classification of 3D Facial Images," in IEEE Access, vol. 10, pp. 23450-23462, 2022, doi: 10.1109/ACCESS.2022.3153357.
