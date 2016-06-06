# Progess

We now have 2 kinds of features:
 * Extracted features from reviews and users
 * Bag of words for each review

The Bag of Words features are reduced using PCA and merged with dense features we.

# Result

| Models             | Mean Absolute Error | R2           |
| -----------------  | ------------------- | ------------ |
| Dummy Regression   | 1.27                | -9.22 * e^-7 |
| Linear Regression  | 1.07                | 0.26         |
| Ridge Regression   | 1.07                | 0.26         |
| Passive Aggressive | 1.42                | -2.10        |
