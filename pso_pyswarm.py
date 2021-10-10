import numpy as np
import pandas as pd
import pyswarms as ps
from scipy.spatial import cKDTree
import time
from datetime import datetime as dt
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,auc,roc_curve,classification_report,explained_variance_score, mean_absolute_error,mean_squared_error,r2_score

def compute_gamma(X,y):
    n_neighbors = 10
    M = X.shape[0]
    # xnik = np.array([])
    ynik = np.array([])
    delta = np.zeros(n_neighbors )
    gamma = np.zeros(n_neighbors )
    tree = cKDTree(X)
    dist, ind = tree.query(X,k=n_neighbors +1, p=2) #ind.shape()=M*self.p

    ind = np.delete(ind,0,axis=1)
    dist = np.delete(dist,0,axis=1)
    # xnik = np.zeros((M*k,p))
    ynik = np.zeros((M,n_neighbors))    
    # xnik = x[ind]
    for row in range(M):
        for col in range(n_neighbors):
            ynik[row,col] = y[ind[row][col]]

    for row in range(M):
        ynik[row,:] -= y[row]

    delta =np.sum(dist**2,axis=0)/M
    gamma = np.sum(ynik**2,axis=0)/(2*M)
    if all(i == 0 for i in delta):
        return 
    else: 
        AandT = np.polyfit(delta,gamma,1)
    return abs(AandT[1])

# Define objective function
def f_per_particle( m):
    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features

    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    alpha = 0.9
    total_features = X.shape[1]
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0: 
        #if the particle subset is only zeros, get the original set of attributes
        X_subset = X
    else:
        X_subset = X[:,m==1]
    particleScore = list()
    particleSize = list()
    score = abs(compute_gamma(X_subset, y))
    particleScore.append(score)
    particleSize.append(X_subset.shape[1])
    # Compute for the objective function
    j = (alpha * (1.0 - score)+ (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x):
    """
    Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, ). The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i]) for i in range(n_particles)]
    #print("f j: ", j)
    return np.array(j)

dataid = 5
if dataid == 0:
    dataset ="puma32h"
elif dataid == 1:
    dataset ="ailerons"
elif dataid == 2:
    dataset ="bank32nh"
elif dataid == 3:
    dataset ="pol"
elif dataid == 4:
    dataset ="triazines"
elif dataid == 5:
    dataset ="parkinsons_updrs"
elif dataid == 6:
    dataset = "Residential-Building-Data-Set"
elif dataid == 7:
    dataset = "slice_localization_data"

data_loc_path = "./regression_dataset/"
location = data_loc_path + dataset + ".csv"
df = pd.read_csv(location,header=0)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
label_column = ['label']
feature_allset = [f for f in df.iloc[:,:-1].columns]
dimensions = len(feature_allset)
print(dimensions)
options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}
start = dt.now()
print("Started at: ", str(start))
optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)
best_cost, best_subset = optimizer.optimize(f, iters=1000,verbose=2)
# bests = optimizer.personal_best_pos #optimizer.get_pos_history

cv = 5
kfold = KFold(n_splits=cv, random_state=123, shuffle=True)
tmp_list = []
tmp_score = 0.0
params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
      'learning_rate': 0.2, 'loss': 'ls'}
for train, test in kfold.split(X, y):
    X_train = X[train,:]
    X_test = X[test,:]
    y_train = y[train]
    y_test = y[test]
    model_gbt = ensemble.GradientBoostingRegressor(**params)
    model_gbt.fit(X_train[:,best_subset==1], y_train)
    y_pre = model_gbt.predict(X_test[:,best_subset==1])
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
print(dataset)
end = dt.now()
print("Finished at: ", str(end))
total = end-start
print("Total time spent: ", total)
