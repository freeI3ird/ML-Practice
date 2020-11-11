# Machine learning and AI

### Concepts
1. What is ML
   - Art and science, of giving computers the ability to make decision using data and without being explicitly programmed.

2. ROC curve
   - A `receiver operating characteristic` curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.
   - The curve is created by plotting the true positive rate against the false positive rate at various threshold settings.
   - ![ROC-Curve](roc-curve-v2.png)
   - If threshold = 0, then TPR = FPR = 1
   - If threshold = 1, then TPR = FPR = 0
   - Random Classifier has Area under curve(AUC) = 0.5.
3. Convergence
   1. When the model “converges” there is usually no significant error decrease / performance increase anymore.
   2. An iterative algorithm is said to converge when, as the iterations proceed, the output gets closer and closer to some specific value. More precisely, no matter how small an error range you choose, if you continue long enough the function will eventually stay within that error range around the final value

### Supervised Learning
1. Feature = predictor variables = independent variables
2. Target variable = dependent variable = label = response variable.

### Classification



### Evaluation Metric

###### Classification
1. Accuracy
2. Precision, Recall, F1-score
3. Area under ROC curve (AUC)

### HyperParameter
1. **HyperParameter**
   - Parameters that specify the model e.g regularization parameter ('lambda'/alpha), No. of layers in neural network, n_neigbhors in KNN.
   - These Paramters can't be learned by fitting the model to data.
2. How to Choose HyperParameter
   - Best approach till date is try all combinations of HyperParameters.

##### HyperParameter Tuning
1. GridSearchCV: Grid Search Cross validation.
   - Say if we have two HyperParameters 'alpha' and 'C', then we will consider all combinations of the possible values of alpha & C.
   - For each combination of values, we will perfrom `k-fold cross validation` to measure the performance of model for that particular HyperParameters combination and choose the best one.
   - Using only train test split would risk overfitting the HyperParameter to the test set.

### Loss function
1. MSE/MAE/HUBER loss function and their sensitivity
   1. https://www.evergreeninnovations.co/blog-machine-learning-loss-functions/#:~:text=The%20MSE%20is%20calculated%20by,errors%20compared%20to%20small%20ones.
2. Gradient Basics
   1. https://betterexplained.com/articles/vector-calculus-understanding-the-gradient/
   2. https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient
   2. Gradient is generalization of derivative
   3. Gradient is a function. INPUT = 3 coordinates as a position, OUTPUT= 3 coordinates as a direction.
   4. The gradient points to the direction of greatest increase; keep following the gradient, and you will reach the local maximum.
