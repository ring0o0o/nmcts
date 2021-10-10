"""
1-step TD prediction
"""
import time
import random
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn import ensemble
from sklearn.svm import SVC,SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score,auc,roc_curve,classification_report,explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt

class state():
    def __init__(self,feature_allset,feature_subset):
        self.feature_allset = feature_allset
        self.feature_subset = feature_subset

    def getPossibleActions(self):
        actions = [i for i in self.feature_allset if i not in self.feature_subset]
        return actions
        
    def next_state(self,action):
        newState = deepcopy(self)
        newState.feature_subset.append(str(action))
        return newState

    def compute_acc(self,X_train,y_train):
        n_neighbors = 10
        X = X_train.values
        y = y_train.values
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

    def isTerminal(self):
        if set(self.feature_subset) == set(self.feature_allset):
            return True
        else:
            return False

    def __str__(self):
        return str(self.feature_subset)
    def __repr__(self):
        return str(self.feature_subset)
    
class Node():
    def __init__(self, state = None, parent = None, depth = 0):
        """
        Initialises a MCTS node
        :param parent: parent node
        :param state: node state
        :param depth: level in tree
        """
        self.parent = parent
        self.state = state
        self.depth = depth
        self.visits = 1
        self.children = []
        self.average_reward = 0.0
        self.total_reward = 0.0

    def expand(self,state):
        """
        Expands the current node
        """
        depth = self.depth + 1
            # new node
        node = Node(parent=self, state=state, depth=depth)
            # add child node
        self.children.append(node)
        return node
 

class Undigraph():
    def __init__(self):
       # self.nodes = {}#key is state.feature_subset, value is the number of visits of the state visited before
        self.act_n = 2
        self.state_values = defaultdict(lambda:50.0)
        self.init_state = state(feature_allset,[])
        self.root = Node(self.init_state)
        #self.nodes.append(self.root)

    def UCB(self,node):
        scaler1 = MinMaxScaler(feature_range=[0,10])
        scaler2 = MinMaxScaler()
        best_children = []
        best_reward = 10000.0
        for child in node.children:
            average_reward_i = scaler1.fit_transform(child.total_reward / child.visits)
            delta_i = scaler2.fit_transform(math.sqrt(2*math.log(node.visits)) / child.visits)
            ucg_score = average_reward_i + delta_i
            if ucg_score < best_reward:
                best_children = child
                best_reward = ucg_score
        return best_children

    def explore_based(self,state):
        actions = state.getPossibleActions()
        f = np.random.choice(actions)
        return f

    def exploit_based(self,node,aor,b):
#   Add a feature f from the repository of state using UCB 
#   or a new feature using AOR ranking
        actions = node.state.getPossibleActions()
        l = list(map(lambda x: int(x), actions))
        minAOR = np.min(aor[1,l])   # index of maximum AOR-value
        t = np.where(aor[1,l] == minAOR)[0]
        actions = np.array(actions)
        f = np.random.choice(actions[t])
        return f

    def update_state_value(self,td_target,state,learning_rate):
        td_error = td_target - self.state_values[tuple(state.feature_subset)]
        self.state_values[tuple(state.feature_subset)] += learning_rate * td_error

    def update_aor(self,aor,f,state):
        k = aor[0,int(f)] #k is visit number of feature f
        aor[1,int(f)] = ((k-1)*aor[1,int(f)] + self.state_values[tuple(state.feature_subset)])/k
    def Search(self,epsiodes,X_train,y_train,X_test,y_test,discount_rate,alpha,epsilon,alpha_decay_rate,epsilon_decay_rate):
        for episode in range(0, epsiodes):
            print('episode=',episode)
            state = self.init_state
            self.node = self.root
            old_reward = []
            i = 0
            flag = True
            while flag and not state.isTerminal():
                i += 1
                if state == self.init_state:
                    flag1 = True
                    while flag1:
                        action = self.explore_based(state)
                        next_state = state.next_state(action)
                        reward = next_state.compute_acc(X_train.loc[:,next_state.feature_subset],y_train)
                        if reward:
                            flag1=False
                    
                    new_node = self.node.expand(next_state)
                    new_node.total_reward += reward
                    self.state_values[tuple(next_state.feature_subset)] = reward

                elif np.random.uniform(0, 1) > (1.0 - epsilon):
                    action = self.explore_based(state)
                    next_state = state.next_state(action)

                    new_node = self.node.expand(next_state)
                    reward = next_state.compute_acc(X_train.loc[:,next_state.feature_subset],y_train)
                    new_node.total_reward += reward
                    td_target = reward + discount_rate * self.state_values[tuple(next_state.feature_subset)]
                    self.update_state_value(td_target, state, alpha)

                else:
                    action = self.exploit_based(self.node,aor,b)
                    next_state = state.next_state(action)

                    new_node = self.node.expand(next_state)
                    reward = next_state.compute_acc(X_train.loc[:,next_state.feature_subset],y_train)
                    new_node.total_reward += reward
                    td_target = reward + discount_rate * self.state_values[tuple(next_state.feature_subset)]
                    self.update_state_value(td_target, state, alpha)
                    

                aor[0,int(action)] += 1
                self.update_aor(aor,action,state)
                self.node = new_node
                state = next_state
                old_reward.append(reward)
                f = 0
                if len(old_reward) > 5:
                    for i in range(len(old_reward)):
                        if old_reward[i] < old_reward[i-1]:
                            f += 1
                        else: f = 0
                        if f == 3:
                            flag = False
            alpha = alpha * alpha_decay_rate
            epsilon = epsilon * epsilon_decay_rate
        return aor

dataid = 3
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
elif dataid == 8:
    dataset = "butterfly"


data_loc_path = "./regression_dataset/"
location = data_loc_path + dataset + ".csv"
data = pd.read_csv(location,header=0)
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
nf = X.shape[1]
label_column = ['label']
feature_allset = [str(f) for f in range(nf) if f not in label_column]
nf = len(feature_allset)
print(nf)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4,shuffle = True)
ss_X=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
X_train = pd.DataFrame(X_train,columns = feature_allset)
X_test = pd.DataFrame(X_test,columns = feature_allset)

epsiodes = 1000 #iteration number
alpha = 0.2 #learning rate
epsilon = 0.5
epsilon_decay_rate = 0.995
alpha_decay_rate = 0.995
regression_flag = True
aor = np.zeros((2,nf))
for i in range(nf):
    aor[1,i] = 10.0
discount_rate = 0.3
m = 3 #stop condition related parameter
b=0.6 #discrete heuristic
start = time.time()
graph = Undigraph()
aor = graph.Search(epsiodes,X_train,y_train,X_test,y_test,discount_rate,alpha,epsilon,alpha_decay_rate,epsilon_decay_rate)
d = np.sqrt(nf)
best_subset = np.argsort(aor[1,:])[:int(d)]
best_subset = [str(i) for i in best_subset]
print('best_subset',best_subset)

if regression_flag:
    cv = 5
    kfold = KFold(n_splits=cv, random_state=123, shuffle=True)
    tmp_list = []
    tmp_list1 = []
    tmp_score = 0.0
    tmp_score1 = 0.0
    params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.2, 'loss': 'ls'}
    score = 0
    for train, test in kfold.split(X, Y):
        X_train = X.loc[train,:]
        X_train.columns = feature_allset
        X_test = X.loc[test,:]
        X_test.columns = feature_allset
        y_train = Y[train]
        y_test = Y[test]

        model_gbt = ensemble.GradientBoostingRegressor(**params)
        model_gbt.fit(X_train[best_subset], Y[train])
        y_pre = model_gbt.predict(X_test[best_subset])
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
    plt.plot(np.arange(X_test.loc[:,best_subset].shape[0]),y_test.values.ravel(), color = 'k', label = 'y')
    plt.plot(np.arange(X_test.loc[:,best_subset].shape[0]),y_pre, color= 'r', label= 'SVR')
    plt.title('SVR_result')
    plt.savefig('result.png')
    #plt.show()

    elapsed = time.time()-start
    print('time=',elapsed)

# clf = SVC(kernel = 'rbf')
# clf.fit(X_train[best_subset],y_train)
# y_pre = clf.predict(X_test[tuple(best_subset)])
# print(classification_report(y_test,y_pre))




