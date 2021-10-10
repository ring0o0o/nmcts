# coding:utf-8
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from fuse import mcts,ReliefF
from sklearn import ensemble
from sklearn.metrics import classification_report,explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
import time

seed = 12345
np.random.seed(seed)

class FeatureSelectionState():
    def __init__(self, Feature_COLUMNS):
        self.Feature_COLUMNS = Feature_COLUMNS
        self.feature_subset = []
        self.search_depth = 20

    def getPossibleActions(self,children):
        possibleActions = []
        for i in self.Feature_COLUMNS:
            if i not in self.feature_subset and i not in children:
                possibleActions.append(i)
        return possibleActions

    def takeAction(self,action):
        newState = deepcopy(self)
        newState.feature_subset.append(str(action))
        return newState

    def isTerminal(self):
        # if len(node.children) == 0:
        if len(self.feature_subset) > self.search_depth:
            return True
        else:
            return False

    def getReward(self,X,y):
        Relief = ReliefF()
        W = Relief.getWeight(X.values,y.values,round(X.shape[0]/3.0))
        return W

    def __str__(self):
        return str(self.feature_subset)
    def __repr__(self):
        return str(self.feature_subset)

  # 1 读数据
dataid = 7
if dataid == 0:
    df = pd.read_csv("puma32h.csv",header = 0)
elif dataid == 1:
    df = pd.read_csv("ailerons.csv",header = 0)
elif dataid == 2:
    df = pd.read_csv("bank32nh.csv",header = 0)
elif dataid == 3:
    df = pd.read_csv("triazines.csv",header = 0)
elif dataid == 4:
    df = pd.read_csv("pol.csv",header = 0)
elif dataid == 5:
    df = pd.read_csv("parkinsons_updrs.csv",header = 0)
elif dataid == 6:
    df = pd.read_csv("Residential-Building-Data-Set.csv",header = 0)
elif dataid == 7:
    df = pd.read_csv("slice_localization_data.csv",header = 0)  

label_column=['label']
Feature_COLUMNS=[f for f in df.columns if f not in label_column]
nf = len(Feature_COLUMNS)
print('ori_columns=',nf)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
std = X_scaled.std(0)
Feature_COLUMNS_new = [x for i,x in enumerate(Feature_COLUMNS) if std[i] != 0]
print('after_coumns=',len(Feature_COLUMNS_new))
X_train = pd.DataFrame(X_scaled[:,np.where(std != 0)[0]],columns = Feature_COLUMNS_new)

start = time.time()

reliefF = ReliefF()
score = reliefF.getWeight(X_train.values,y.values)
sorted_COLUMNS = []
for i in np.argsort(score)[::-1]:
    sorted_COLUMNS.append(Feature_COLUMNS_new[i])
    
X_train_sorted = X_train.loc[:,sorted_COLUMNS]
initialState = FeatureSelectionState(sorted_COLUMNS)
mcts = mcts(iterationLimit=1000,X=X_train_sorted,y=y)
best_subset = mcts.search(initialState=initialState)
print('best_subset=',best_subset)

cv = 5
kfold = KFold(n_splits=cv, random_state=123, shuffle=True)
tmp_list = []
tmp_score = 0.0
start = time.time()
params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
      'learning_rate': 0.2, 'loss': 'ls'}
score = 0

for train, test in kfold.split(X, y):
    X_train = X.loc[train,:]
    X_test = X.loc[test,:]
    y_train = y.loc[train]
    y_test = y.loc[test]

    model_gbt = ensemble.GradientBoostingRegressor(**params)
    model_gbt.fit(X_train.loc[:,best_subset], y_train)
    y_pre = model_gbt.predict(X_test.loc[:,best_subset])
    # model_gbt = SVR(kernel='rbf',C=80, gamma = 1/len(best_subset)) # 建立支持向量机回归模型对象, C=1e3, gamma = 0.
    # model_gbt.fit(X_train[best_subset],y_train)
    # y_pre = model_gbt.predict(X_test[best_subset])
    params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.2, 'loss': 'ls'}
    #R-Squared 越大，表示模型拟合效果越好。
    model_metrics_name = [explained_variance_score, mean_absolute_error,mean_squared_error,r2_score]
    for mdl in model_metrics_name:
        tmp_score = mdl(y_test, y_pre)
        tmp_list.append(tmp_score)
tmp_list = np.array(tmp_list).reshape(cv,-1).sum(axis = 0)/cv
df = pd.DataFrame(tmp_list,columns = ['model_gbt'], index = ['ev','mae','mse','r2'])
print(df)

elapsed = time.time()-start
print('time=',elapsed)
