# USC DSCI553 Spring 2021

This repository stores all my submitted answers for HWs.

Note that for each assignment, a maximum of 0.7 bonus point could be earn by submitting Scala point.

On the other hand for competition project, any RMSE score lower than 0.98 would get full 8 points. The top six students in the class would earn extra points on top of that.

Here are the winners of this semester's competition project and their RMSE score on test data:

1. Zeyang Gong (0.9721)

2. Jiemin Tang (0.9744)

3. Nitin Chandra Perumandl (0.9745)

4. Shiyang Chen (0.9749)

5. Matheus Schmitz (0.9750)

6. Yuxin Jiang (0.9752)


| | Topics | Score |
| ------------- | ------------- | ------------- |
| HW1  | Spark  | 7.6 / 7 |
| HW2  | Frequent itemsets | 7.55 / 7 |
| HW3 | Recommender system  | 6 / 7 |
| HW4  | Graph | 5.7 / 7 |
| HW5  | Stream  | 7.7 / 7 |
| HW6  | Clustering | 7.7 / 7 |
| Competition | Recommender system | 8 / 8 (RMSE:0.9762 on test data)|

# Competition Project

To improve the recommender system from HW3, I first decided to use model-based CF as the primary recommender for its better performance over item-based CF.
First I tried to improve model-based CF by adding more features.

One thing I found out is that feature engineering is very important. For example, I first use one hot encoding to add "categories" of business.json.
However, there are 1000+ different categories. As a result, adding those dummy variables directly to the model will slow down the training process and did not improve the model's performance. So I used PCA to conduct dimensionality reduction and only kept the top 10 pca components as the features. This resulted in about 0.002 improvement in RMSE.

On the other hand, I tried to utilize the power of item-based CF by implementing a feature-augmented CF. I did so by adding the prediction as a feature to the model-based CF. And that is the best model I had up to the deadline.

Other recommender systems I tried but wasn't very successful:
Use K-means to cluster the data and train model-based CF on each cluster (make predictions by first assign active user-business pair to nearest cluster)
&#8594 One possible reason why this method is not working is because to do K-means clustering, all the points are naively represented by a point in Euclidean space
For certain type of business (mainly restaurants), collect extra features for these businesses (e.g., GoodforKids) and train two models: one for that type of businesss and one for all businesses
--> This method is not working probably because the exclusion of some valuable information from other businesses in the first model (which is about 1/4 of the training data)

Error Distribution:
>=0 and <1: 105773
>=1 and <2: 34123
>=2 and <3: 6310
>=3 and <4: 790
>=4: 0
