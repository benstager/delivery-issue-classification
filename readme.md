SHIPMENT ISSUE MODEL AND OPTIMIZATION WRITEUP
<br><br>
**File navigation**:

*classfication_task.csv* - labeled dataset

*classification_test.csv* - out of sample predictions

*col_df* - column metadata to specify ML model which columns to use

*final_model.pkl* - trained SVC model

*model_final.ipynb* - interactive jupyter notebook to show development process

*model_final.py* - model code that runs and generates predictions into pandas dataframe

*model_research.ipynb* - essentially scratch paper for my development process

*out_of_sample_predictions* - predictions made on unlabeled dataset

*requirements.txt* - file to install if you don't have the corresponding requirements

**Introduction**

When dealing with product sales, demand, and logistics, one must be cautious to evaluate all factors to ensure that the problem at hand
is solved correctly and efficiently. The problem given was multi-fold: correctly fit a binary classifier to a dataset that prescribes
shipment issues correlated to a number of categorical and continuous attributes, predict a set of out-of-sample observations, and
create an optimization algorithm to determine which of those potential problematic shipments should be processed using expedited shipping.
In the following writeup, I will outline my methodology for each of these steps and their corresponding results.

**Binary classification model and results**

To build my classification algorithm, I began by naturally examining each feature I had - a significant majority of them being 
categorical variables. Amongst these categorical variables, a majority of those followed an ordinal pattern - i.e. Medium, Large
package size - signaling that an ordinal encoding would be most appropriate. Variables such as location, weather, and shipment method were preprocessed using ordinal encoding in order to preserve the order that such variables preserve. Variables like carrier were encoded using One-Hot, as I do not retain enough external information to make a judgement call if they are correlated ordinally towards a certain class. The lone continuous feature, distance, was preprocessed using routine standardization scaling The response variable, shipment_issue, was considerably imbalanced, with an approximately 90-10 split in favor of the negative class (no shipment issue). To alleviate an incorrect decision boundary being established, I used undersampling of the majority class so that each algorithm would see enough of both the 0 and 1 class. My philosophy when creating any model, one that I believe is held by many machine learning practioners, is to start simple and use all the data, and then refine. Before any chi-squared, collinearity test, or RFE procedure was performed, I wanted to see how the models would perform ingesting all features. To train, I used a stratified KFold cross validation methodology. Since I am resampling the data to a 50 50 response variable split, this means that my 'baseline model' would yield an ROC-AUC of 50 pct. As I am essentially comparing apples to apples, I felt it appropriate to not resample the test set back into the majority split, although this is often done. I employed my 3 favorite (and most common) classification frameworks - sklearn's Logistic Regression and Random Forest, and xgboost's XGBClassifier. If these 3 methods performed poorly, it would be futile to try and fit
a more complex framework like a feed forward neural network.

Before fitting any models, I decided to do some slight EDA, even given the time constraints. By examaning the samples that had a shipment issue, we can see that those with issues have some interesting feature distributions. Specifically - 61% of total samples
used carrier_A, but 78% of shipments with issues used carrier_A. 50% of total samples were medium packages, but 58% of those
with issues were large packages. 33% of total samples were Urban locations, but close to 50% of those with issues were Urban locations.
A lot of these were potentially obvious, but it is important to note this. Once I finished encoding all necessary variables,
I started my training process. My general procedure was to set aside 80% of samples for traning and validation, and then the 
remaning 20% to report test error. To train the models, I used 5-fold cross validation for both Random Forest and Support Vector
Machine. I stratified sampling for each validation set, but not for training. I did this in effort to reduce an incorrect decision
boundary being generated, and error definitely improved by doing so. The model would be trained and predicted a total of 5 times, and then the validation error reported would be the average of the 5 ROC-AUC scores. For a baseline model, I generated a vector of 0s (the majority class), that was of length of the test set. This would demonstrate that the baseline ROC-AUC (integral of the ROC curve) would be .5 (no discrimination from random choice). I began by using Random Forest (with Optuna for hyperparameter tuning), but found myself
frustrated that I was only getting about a .58 ROC-AUC. I then decided to use a Support Vector Machine with a radial basis function kernel. Initially I was resistant to using SVM due to its complication with dimensionality increases, but found its performance to be much better. My SVM algorithm generated a validation ROC-AUC of .7, and a test set error of .66. I started with features that were
obvious to use (package_type, pop_density), and added more until I saw a regression in ROC-AUC. When predicting on the out of sample
data, I generated a distribution of 4

**Optimization problem**

Creating mathematical optimization problems can be exceptionally tricky. To devise this optimization problem, I will only consider
samples that have predicted a potential shipment issue. Of the 128 potential shipment issues that were predicted, around
25% of them dictated expedited shipping. To create an optimization function, we can define a total cost function using a number of 
different variables. Suppose we define the binary optimization problem using the following variables 

$$
f(x_1, ..., x_n) = \sum_{i = 0}^n [c  \cdot h_i \cdot x_i + d \cdot (1 - x_i) \cdot p_i]
$$

Where:
- \( c \) is the cost of expedited shipping.
- \( d \) is the cost factor for regular shipping based on probability.
- \( x_i \) is a binary decision variable, where \( x_i = 1 \) means item \( i \) is shipped expedited, and \( x_i = 0 \) otherwise.
- \( p_i \) is the probability that item \( i \) would need a replacement.
- \( h_i \) is the shipping distance cost assosciated with that sample.

We can define a constraint of total cost being under a certain threshold, such that:

$$
\sum_{i} c \cdot x_i \leq B
$$

Where:
- \( c \) is the cost of expedited shipping.
- \( x_i \) is a binary decision variable, where \( x_i = 1 \) means item \( i \) is shipped expedited, and \( x_i = 0 \) otherwise.
- B is the total cost you are willing to incur on expedited shipping

The resulting function is a multivariate function that would require a Python linear programming function, such as PuLP. The
returned data from this problem would be labels $x_i \in \{0,1\}$ that prescribe if the replacement order should be sent or not.