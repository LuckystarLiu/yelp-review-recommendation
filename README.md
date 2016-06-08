# Yelp Reviews Recommendation

### Members

_Listed in alphabetically order_

Desheng Liu (dli848): deshengliu2015@u.northwestern.edu

Shiping Zhao (szd041): shipingzhao2015@u.northwestern.edu

Yan Liu (ylk070): yanliu2015@u.northwestern.edu

### Organization

This is a Machine Learning project in EECS 349 at Northwestern University.

### Synopsis

The reviews we would see in the Yelp app in a first glance are mainly 2 kinds: reviews with high votes, and reviews that are fresh. Although the reviews with high votes could be useful, they are probably out-of-date. Thus, fresh reviews are also needed to be displayed in the reviews flow. However, most of the fresh reviews are low quality, or say, useless. It would be better to rank those more useful reviews to be higher than less useful ones, so that the users could read the useful ones first to save their time.

To rank the reviews in terms of their quality (weighted by kinds of votes and normalized by post time), the learners we tried to use are regression learners. To be more specific, we used Linear Regression, Ridge Regression, Passive Aggressive Regression, AdaBoost, Random Forest, etc. In terms of features, we used both text features (bag of words) and metadata of reviews and their users. In addition, the dimentsion of the text features are normalized by overall frequency and reduced using PCA technique.

The dataset we used are the json files achieved from [Yelp's official website](https://www.yelp.com/dataset_challenge). We first preprocessed them to be standardized csv files. Then trained and tested them with scikit-learn library. To measure the success, we used the Dummy Regressor learner as the baseline. It is similar to the ZeroR classifier in Weka which simply predicts the result by assigning a simple value (i.e. mean, median, constant quality). We compared the performaces of different kinds of regression learners with the baseline learner.

We evaluated the performance of regression learners with R2 score. It could be negative because the prediction for regression problem could be arbitrarily bad. And the closer to 1, the better it is. For the result, the R2 score of the __baseline learner is -9.22 * 10^-7__. The __best learner we find is Random Forest, earning 0.48 R2 score__. The second best ones are Linear and Ridge Regression, both getting 0.26 R2 score. The key features that greatly improved our result are metadata of reviews and users, such as votes, stars, etc. When using only text features, we only get 0.02 R2 score with Linear Regression and 0.03 with Ridge Regression.

### Performance

Features: Bag of Words

| Models             | Mean Absolute Error | R2           |
| -----------------  | ------------------- | ------------ |
| Dummy Regression   | 1.35                | -0.008       |
| Linear Regression  | 1.32                | 0.02         |
| Ridge Regression   | 1.28                | 0.03         |
| Passive Aggressive | 1.36                | -0.08        |
| AdaBoost           | 4.74                | -3.71        |
| Random Forest      | running             | running      |

Features: Original Features + Bag of Words

| Models             | Mean Absolute Error | R2           |
| -----------------  | ------------------- | ------------ |
| Dummy Regression   | 1.27                | -9.22 * e^-7 |
| Linear Regression  | 1.07                | 0.26         |
| Ridge Regression   | 1.07                | 0.26         |
| Passive Aggressive | 1.42                | -2.10        |
| AdaBoost           | 15.17               | -23.69       |
| Random Forest      | 0.98                | 0.48         |
