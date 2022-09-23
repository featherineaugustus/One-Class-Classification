# One-Class-Classification

Data imbalance is a serious data science problem that one can encounter. It is even more imbalanced in the context of anormaly detection, where the anormaly class can be extremely little as compared to the majority class. In the most extreme case, the data for the anormaly class may not even be available. In such situation, it is not feasible to train a binary classifier to identify those anormalies. Instead, we can deploy one-class classifiers.

We tried 4 different one-class classifier on a simple dataset, to illustrate the power of one-class models to identify anormalies.

1. One-Class Support Vector Machines (SVM)
2. Isolation Forest
3. Minimum Covariance Determinant
4. Local Outlier Factor

Generally, it appears that one-class SVM works the best. However, we did not perform gridsearch for hyperparameters optimization. Then again, it may not be possible to optimize the hyperparameters as the anormaly class are not available for training.

The original work can be found at:

https://machinelearningmastery.com/one-class-classification-algorithms/
