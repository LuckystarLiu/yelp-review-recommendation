# Features

We now have 2 kinds of features:
 * Original Features: features recorded in the original reviews and users
 * Bag of words: normalized frequency of words in each review

The dimension of the Bag of Words feature is reduced to 100 using PCA.

# Result

Features: Bag of Words

| Models             | Mean Absolute Error | R2           |
| -----------------  | ------------------- | ------------ |
| Dummy Regression   | 1.35                | -0.008       |
| Linear Regression  | 1.32                | 0.02         |
| Ridge Regression   | 1.28                | 0.03         |
| Passive Aggressive | 1.36                | -0.08        |

Features: Original Features + Bag of Words

| Models             | Mean Absolute Error | R2           |
| -----------------  | ------------------- | ------------ |
| Dummy Regression   | 1.27                | -9.22 * e^-7 |
| Linear Regression  | 1.07                | 0.26         |
| Ridge Regression   | 1.07                | 0.26         |
| Passive Aggressive | 1.42                | -2.10        |
| AdaBoost           | 15.17               | -23.69       |
| Random Forest      | 0.98                | 0.48         |
