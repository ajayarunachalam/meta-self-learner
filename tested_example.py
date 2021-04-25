__author__ = 'Ajay Arunachalam'
__version__ = '0.0.1'
__date__ = '25.4.2021'

# load libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from tabulate import tabulate
from xgboost import XGBClassifier

# load built package functions
from msl.MetaLearning import *
from plot_metric.functions import MultiClassClassification
from msl.cf_matrix import make_confusion_matrix

#fixing random state
random_state=123

# Load dataset (we just selected 4 classes of digits)
X, Y = load_digits(n_class=4, return_X_y=True)

print(f'Predictors:')
print(f'{X}')

print(f'Outcome:')
print(f'{Y}')

# Add noisy features to make the problem more harder
random_state = np.random.RandomState(123)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 1000 * n_features)]

## Spliting data into train and test sets.
X, X_test, y, y_test = train_test_split(X, Y, test_size=0.2, 
                                        random_state=123)
    
## Spliting train data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, 
                                                      random_state=1)

print('Data shape:')
print('X_train: %s, X_valid: %s, X_test: %s \n' %(X_train.shape, X_valid.shape, 
                                                  X_test.shape))

# Create list to store logloss of individual classifiers (single classifier) & meta self-learners
ll_sc, ll_ensemble1, ll_ensemble2, ll_ensemble1_cc, ll_ensemble2_cc, ll_ensemble3, ll_lr, ll_gb = [[] for i in range(8)]

#Defining the classifiers
clfs = {'LR'  : LogisticRegression(random_state=random_state), 
        'SVM' : SVC(probability=True, random_state=random_state), 
        'RF'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                       random_state=random_state), 
       'GBM' : GradientBoostingClassifier(n_estimators=50, 
                                          random_state=random_state), 
        'ETC' : ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
                                     random_state=random_state),
        'KNN' : KNeighborsClassifier(n_neighbors=30)}
    
#predictions on the validation and test sets
p_valid = []
p_test = []

########################### LAYER 1 ##########################
'''
First layer (individual classifiers)
All classifiers are applied twice:
Training on (X_train, y_train) and predicting on (X_valid)
Training on (X, y) and predicting on (X_test)
We can add / remove classifiers or change parameter values to see the effect on final results.
'''
print('Performance of individual classifiers (1st layer) on X_test')   
print('------------------------------------------------------------')

for lg, clf in clfs.items():
    #First run. Training on (X_train, y_train) and predicting on X_valid.
    clf.fit(X_train, y_train.ravel())
    yv = clf.predict_proba(X_valid)
    p_valid.append(yv)

    # second run. Training on (X, y) and predicting on X_test.
    clf.fit(X, y.ravel())
    yt= clf.predict_proba(X_test)
    p_test.append(yt)

    # print the performance for each classifier
    print('{:10s} {:2s} {:1.7f}'. format('%s:' %(lg), 'logloss =>', log_loss(y_test, yt)))
    #Saving the logloss score
    ll_sc.append(log_loss(y_test, yt)) #Saving the logloss score
print('')

# Configure the number of class to input into the model

NUM_CLASS = MetaEnsemble.set_config(NUM_CLASS=4) # Enter your number of classes in the dataset here

# Using Ensemble1 and Ensemble2 in a THREE-LAYERED META LEARNER architecture.

########################### LAYER 2 ##########################
'''
(optimization based ensembles)
Predictions on X_valid are used as training set (XV) and predictions on X_test are used as test set (XT). 
Ensemble1, Ensemble2 and their calibrated versions are applied.
'''
print('Performance of optimization based meta self-learners (2nd layer) on X_test')
print('------------------------------------------------------------')
#Creating the data for the 2nd layer.

XV = np.hstack(p_valid)
XT = np.hstack(p_test)

# Ensemble1

en1 = MetaEnsemble.Ensemble_one(NUM_CLASS) # as we have 26 classes n_classes=26
en1.fit(XV, y_valid.ravel())
w_en1 = en1.w
y_en1 = en1.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Ensemble1:', 'logloss =>', log_loss(y_test, y_en1)))
ll_ensemble1.append(log_loss(y_test, y_en1)) #Saving the logloss score

#Calibrated version of Ensemble1

cc_en1 = CalibratedClassifierCV(en1, method='isotonic')
cc_en1.fit(XV,y_valid.ravel())
y_cc_en1 = cc_en1.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_Ensemble1:', 'logloss =>', log_loss(y_test, y_cc_en1)))
ll_ensemble1_cc.append(log_loss(y_test, y_cc_en1)) #Saving the logloss score

# Ensemble2

en2 = MetaEnsemble.Ensemble_two(NUM_CLASS) # as we have 26 classes n_classes=26
en2.fit(XV,y_valid.ravel())
w_en2 = en2.w
y_en2 = en2.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Ensemble2:', 'logloss =>', log_loss(y_test, y_en2)))
ll_ensemble2.append(log_loss(y_test, y_en2)) #Saving the logloss score

#Calibrated version of Ensemble2

cc_en2 = CalibratedClassifierCV(en2, method='isotonic')
cc_en2.fit(XV,y_valid.ravel())
y_cc_en2 = cc_en2.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_Ensemble2:', 'logloss =>', log_loss(y_test, y_cc_en2)))
ll_ensemble2_cc.append(log_loss(y_test, y_cc_en2)) #Saving the logloss score
print('')

############# Third layer (weighted average) ######################################
# Simple weighted average of the previous 4 predictions.
print('Performance of agggregation of the self-learners (3rd layer) on X_test')
print('------------------------------------------------------------')
y_thirdlayer = (y_en1 * 4./9.) + (y_cc_en1 * 2./9.) + (y_en2 * 2./9.) + (y_cc_en2 * 1./9.)
print('{:20s} {:2s} {:1.7f}'.format('3rd_layer:', 'logloss =>', log_loss(y_test, y_thirdlayer)))
ll_ensemble3.append(log_loss(y_test, y_thirdlayer))

'''
# Plotting the weights of each ensemble
In the case of Ensemble1, there is a weight for each prediction 
and in the case of Ensemble2 there is a weight for each class for each prediction.
'''
from tabulate import tabulate
print(' Weights of Ensemble1:')
print('|---------------------------------------------|')
wA = np.round(w_en1, decimals=2).reshape(1,-1)
print(tabulate(wA, headers=clfs.keys(), tablefmt="orgtbl"))
print('')
print(' Weights of Ensemble2:')
print('|-------------------------------------------------------------------------------------------|')
wB = np.round(w_en2.reshape((-1,NUM_CLASS)), decimals=2) # 26 is no. of classes (NUM_CLASS)
wB = np.hstack((np.array(list(clfs.keys()), dtype=str).reshape(-1,1), wB))
print(tabulate(wB, headers=['y%s'%(i) for i in range(NUM_CLASS)], tablefmt="orgtbl"))

'''
Comparing the ensemble results with sklearn LogisticRegression based stacking of classifiers.
Both techniques Ensemble1 and Ensemble2 optimizes an objective function. 
In this experiment I am using the multi-class logloss as objective function. 
Therefore, the two proposed methods basically become implementations of LogisticRegression. The following
code allows to compare the results of sklearn implementation of LogisticRegression with the proposed ensembles.
'''
#By default the best C parameter is obtained with a cross-validation approach, doing grid search with
#10 values defined in a logarithmic scale between 1e-4 and 1e4.
#Change parameters to see how they affect the final results.

# LogisticRegression
lr = LogisticRegressionCV(Cs=10, dual=False, fit_intercept=True,
    intercept_scaling=1.0, max_iter=100,
    multi_class='ovr', n_jobs=1, penalty='l2',
    random_state=random_state,
    solver='lbfgs', tol=0.0001)

lr.fit(XV, y_valid.ravel())
y_lr = lr.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Logistic_Regression:', 'logloss =>', log_loss(y_test, y_lr)))
ll_lr.append(log_loss(y_test, y_lr)) #Saving the logloss score
print('')

'''
Comparing the ensemble results with sklearn GradientBoost based stacking of classifiers.
Both techniques Ensemble1 and Ensemble2 optimizes an objective function. 
In this experiment I am using the multi-class logloss as objective function. 
'''
from xgboost import XGBClassifier
# Gradient boosting
xgb = XGBClassifier(max_depth=5, learning_rate=0.1,n_estimators=10000, objective='multi:softprob',seed=random_state)

# Computing best number of iterations on an internal validation set
XV_train, XV_valid, yv_train, yv_valid = train_test_split(XV, y_valid, test_size=0.15, random_state=random_state)
xgb.fit(XV_train, yv_train, eval_set=[(XV_valid, yv_valid)],
        eval_metric='mlogloss',
        early_stopping_rounds=15, verbose=False)
xgb.n_estimators = xgb.best_iteration
xgb.fit(XV, y_valid.ravel())
y_gb = xgb.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Gradient_Boost:', 'logloss =>', log_loss(y_test, y_gb)))
ll_gb.append(log_loss(y_test, y_gb)) #Saving the logloss score
print('')

print(f'Log-Loss for Base Learners:')
print(f'{ll_sc}')

# Comparison of the 3L ENSEMBLE techniques (Ensemble1, Ensemble2, Ensemble3) with Ensemble Logistic & Ensemble XGBOOST (plotting the results)
#classes = 4

ll_sc = np.array(ll_sc).reshape(-1, len(clfs)).T
#print(ll_sc)
ll_ensemble1 = np.array(ll_ensemble1)
ll_ensemble2 = np.array(ll_ensemble2)
ll_ensemble3 = np.array(ll_ensemble3)
ll_ensemble1_cc = np.array(ll_ensemble1_cc)
ll_ensemble2_cc = np.array(ll_ensemble2_cc)
ll_lr = np.array(ll_lr)
ll_gb = np.array(ll_gb)


plt.figure(figsize=(10,10))
plt.plot(ll_sc, color='black', label='Single_Classifiers')

for i in range(1, 6):
    plt.plot(ll_sc[i], color='black')
plt.title('Log-loss of the different models.')
plt.xlabel('Testing on LDIGITS DATASET with only 4 classes')
plt.ylabel('Log-loss')
plt.grid(True)
plt.legend(loc=1)
plt.show()

plt.title('Log-loss of the different models.')
plt.xlabel('Testing on DIGITS DATASET with only 4 classes')
plt.ylabel('Log-loss')
plt.plot(ll_lr, 'bo-', label='EN_LogisticRegression', )
plt.plot(ll_gb, 'mo-', label='EN_XGBoost')
plt.plot(ll_ensemble1, 'yo-', label='Ensemble1')
plt.plot(ll_ensemble1_cc, 'ko-', label='Calibrated Ensemble1')
plt.plot(ll_ensemble2, 'go-', label='Ensemble2')
plt.plot(ll_ensemble2_cc, 'ko-', label='Calibrated Ensemble2')
plt.plot(ll_ensemble3, 'ro-', label='Ensemble_3rd_layer')

plt.grid(True)
plt.legend(loc=1)
plt.show()

print(np.argmax(y_thirdlayer, axis=1))
y_pred_meta_self_learner = np.argmax(y_thirdlayer, axis=1)

print(f'Predictions from the final layer:')
print(f'{y_pred_meta_self_learner}')

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred_meta_self_learner)
print(cf_matrix)
make_confusion_matrix(cf_matrix, figsize=(8,6), cbar=False, title='Confusion Matrix')

# Visualisation of plots
mc = MultiClassClassification(y_test, y_thirdlayer, labels=[0, 1, 2, 3])
plt.figure(figsize=(13,4))
plt.subplot(131)
mc.plot_roc()
plt.subplot(132)
mc.plot_confusion_matrix()
plt.subplot(133)
mc.plot_confusion_matrix(normalize=True)

plt.savefig('figures/images/plot_multi_classification.png')
plt.show()

mc.print_report()

