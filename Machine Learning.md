
@ Puran Zhang<br>

A computer program is said to learn from experience E with repect to some task T and some performance measure P, if it's performance on T, as measured by p, improves with experience E<br>

## Machine Learning workflow
1. Data Processing (Data Clean, Separate training and test sets, Feature)
2. Trying appropriate algorithms (No Free Lunch)
3. Fitting model parameters
4. Tuning impactful hyperparameters
5. Proper performance metrics
6. Systematic cross-validation
***
# Trying ML Algorithms:
### 1. Supervised Learning 
Given data set, knew what correct output should looks like ( $\exists$ relationship between input and output)<br><br>
**1.1 Regression**: predict results within a continuous output <br>
**1.2 Classification**: predict results in a discrete output <br>

### 2. Unsupervised Learning
No idea what results looks like, derive structure by clustering data (no feedback based on prediction)<br><br>
**2.1 Clustering** <br>
**2.2 Non-clustering**: cocktail party algorithm allows you to find structure in a chaotic environment<br>
### 3. Others: Reinforcement Learning, Recommend Systems

## Linear Regression


## K-Nearest Neighbors (KNN)
a non-parametric, lazy learning method used for classification and regression. The output based on the majority vote (for classification) or mean (or median, for regression) of the k-nearest neighbors in the feature space.<br>

**non-parametric**: it does not make any assumption of the data distribution.

**lazy learning**: a learning method in which generalization of the training data is delayed until a query is made to the system. i.e. there is no explicit training stage, or it is very minimal, which also means that training is very fast in KNN.

3 steps for KNN:<br>
1. Calculate distance (e.g. Euclidean distance, Hamming distance, etc.)
2. Find k closest neighbors
3. Vote for labels or calculate the mean

***
## Overfitting
The model doesn't generalize well from the training data to unseen data<br>
### Signal vs. Noise
**Signal**: the true underlying pattern that you wish to learn from the data in predictive modeling.<br>
**Noise**: the irrelevant information or randomness in a dataset.<br><br>
A well functioning ML algorithm will separate the signal from the noise.This overfit model will then make predictions based on that noise.
### Goodness of fit
In statistics, goodness of fit refers to how closely a model’s predicted values match the observed (true) values.

### Bias-Variance  Tradeoff

Expected squared error at a point x (total error):<br>
$$Err(x) = {Bias}^2 + Variance + {Irreducible} \ {Error}$$
$$ E[(Y - \hat{f}(x))^2] = (E[\hat{f}(x)-f(x)])^2 + E[(\hat{f}(x)-E[\hat{f}(x)])^2] + \sigma^2_e$$
[proof](http://www.inf.ed.ac.uk/teaching/courses/mlsc/Notes/Lecture4/BiasVariance.pdf)

`Bias` is the difference between the average prediction of our model and the correct value which we are trying to predict. <br>

*Model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.* 

`Variance` is the variability of model prediction for a given data point or a value which tells us spread of our data.<br>

*Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before. As a result, such models perform very well on training data but has high error rates on test data.*

![image](m.1.1.png)

### How to detect overfitting

we can split our initial dataset into separate training and test subsets.<br><br>
If our model does much better on the training set than on the test set, then we’re likely overfitting.<br><br>
**Occam’s razor test**: If two models have comparable performance, then you should usually pick the simpler one. (start with a very simple model to serve as a benchmark)

### How to prevent overfitting

#### 1. cross valication
Use your initial training data to generate multiple mini train-test splits. Use these splits to tune your model.<br>
*e.g.we want to split the training set to A (80% of data) and B (20% of data). We then train our model based on A and test the model on B, since we know the ground truth of B now. B is called the validation set. We could use the validation set to tune our parameters* 

`k-fold cross-validation`: we partition the data into k subsets, called folds. Then, we iteratively train the algorithm on k-1 folds while using the remaining fold as the test set (called the “holdout fold”)<br>

*final accuracy = average (k=1, k=2,...)*
![image](m.1.2.png)

`Leave-one-out cross-validation`: $K =$ number of training set<br> since we only test on one sample each time and use the rest of the data to train<br>

Cross-validation allows you to tune hyperparameters with only your original training set. This allows you to keep your test set as a truly unseen dataset for selecting your final model.<br>
#### 2. data augmentation
![image](m.1.3.png)

#### 3. remove feaures
Some algorithms have built-in feature selection.For those that don’t, you can manually improve their generalizability by removing irrelevant input features.<br>
<br>
An interesting way to do so is to tell a story about how each feature fits into the model.<br><br>
**Dropout**: Randomly delete neurons<br>
![image](m.1.5.png)

#### 4. early stopping  (mostly in deep learning)<br>
![image](m.1.4.png)

#### 5. regularization (prefered for classification)
Regularization refers to a broad range of techniques for artificially forcing your model to be simpler.<br><br>
The method will depend on the type of learner you’re using. For example, you could prune a decision tree, use dropout on a neural network, or add a penalty parameter to the cost function in regression.<br><br>
Oftentimes, the regularization method is a hyperparameter as well, which means it can be tuned through cross-validation<br>

#### 6. ensembling
Ensembles are machine learning methods for combining predictions from multiple separate models.<br><br>
**Bagging** reduce the chance overfitting complex models<br>
* It trains a large number of "strong" learners in parallel
* A strong learner is a model that's relatively unconstrained
* Bagging then combines all the strong learners together in order to "smooth out" their predictions<br>

**Boosting** improve the predictive flexibility of simple models<br>
* It trains a large number of "weak" learners in sequence.
* A weak learner is a constrained model (i.e. you could limit the max depth of each decision tree).
*  Each one in the sequence focuses on learning from the mistakes of the one before it.
* Boosting then combines all the weak learners into a single strong learner.


***
# Tuning impactful hyperparameters


```python

```
