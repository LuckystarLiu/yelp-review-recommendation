# Progess
We now have 2 kinds of features
* Extracted features from reviews and users (dense)
* Bag of words for each review (sparse)

The challenge is the bag of words is sparse and could be too large to output to csv. There are basically 3 ways to deal with the combined features:
* Do dimension reduction on sparse data to make it dense and then combine them
* Add those few dense features to the sparse matrix by using scipy's `hstack`
* Create models for sparse features and dense features separately and combine their predictions. 

Accuracy: (1) < (2) < (3)

