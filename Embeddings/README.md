# Low Dimensional Embeddings

The mat file in the GE folder contains two 5 X 7 dimensional cells, preds_train and preds_test, which are embeddings of the train and test set. data is ith row contains embeddings of the ith data fold (we use 5-fold cross validation), and column j contains data for the jth segment from the hierarchichal facial segmentation.

The mat file in the PCA folder contains a 5 X 7 dimensional cells. predicted_train, predicted_test, grps_train, and grps_test are the four fileds of each cell, containing embeddings and group labels of train and test set. Similar to the GE cell data format, the ith row contains embeddings of the ith data fold, and jth column contains data for the jth segment of the hierarchichal facial segmentation.

group_ref is the reference cell array. The name in ith cell indicates the name of the syndrome with group label i (grps=i in the classification code)