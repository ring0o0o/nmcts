import numpy as np
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import time
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn import ensemble
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score,auc,roc_curve,classification_report,explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
import feature_selection as FS

# Loading the diabetes dataset
data = pd.read_csv('slice_localization_data.csv',header = 0)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
nf = X.shape[1]
label_column = ['label']
featnames = [str(f) for f in range(nf) if f not in label_column]
nf = len(featnames)
print(nf)


# Example 3: use Isotropic Selector to perform a complete analysis of the input
# space, recongising relevant, redundant, irrelevant features
start = time.time()
IsotropicSelector = FS.Isotropic_selector(bandwidth = 'rule-of-thumb')
best_subset = IsotropicSelector.feat_selection(X, y.ravel(), feature_names=featnames, strategy ='ffs')
print('Time to complete the feature selection [s]: ' + str(time.time() - start))

cv = 5
kfold = KFold(n_splits=cv, random_state=123, shuffle=True)
tmp_list = []
tmp_list1 = []
tmp_score = 0.0
tmp_score1 = 0.0
start = time.time()
params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
     'learning_rate': 0.2, 'loss': 'ls'}
score = 0
for train, test in kfold.split(X, y):
   X_train = X[train,:]
   X_test = X[test,:]
   y_train = y[train]
   y_test = y[test]

   model_gbt = ensemble.GradientBoostingRegressor(**params)
   model_gbt.fit(X_train[:,best_subset], y[train])
   y_pre = model_gbt.predict(X_test[:,best_subset])
   # model_gbt = SVR(kernel='rbf',C=80, gamma = 1/len(best_subset)) # 建立支持向量机回归模型对象, C=1e3, gamma = 0.
   # model_gbt.fit(X_train[best_subset],y_train)
   # y_pre = model_gbt.predict(X_test[best_subset])
   params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
         'learning_rate': 0.2, 'loss': 'ls'}
   model_gbt1 = ensemble.GradientBoostingRegressor(**params)
   model_gbt1.fit(X_train, y_train)
   y_pre1 = model_gbt1.predict(X_test)
   #R-Squared 越大，表示模型拟合效果越好。
   model_metrics_name = [explained_variance_score, mean_absolute_error,mean_squared_error,r2_score]
   for mdl in model_metrics_name:
       tmp_score = mdl(y_test, y_pre)
       tmp_score1 = mdl(y_test, y_pre1)
       tmp_list.append(tmp_score)
       tmp_list1.append(tmp_score1)
tmp_list = np.array(tmp_list).reshape(cv,-1).sum(axis = 0)/cv
tmp_list1 = np.array(tmp_list1).reshape(cv,-1).sum(axis = 0)/cv
df = pd.DataFrame(tmp_list,columns = ['model_gbt'], index = ['ev','mae','mse','r2'])
print(df)
df1 = pd.DataFrame(tmp_list1,columns = ['model_gbt1'], index = ['ev','mae','mse','r2'])
print(df1)
