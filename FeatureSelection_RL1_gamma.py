#!/usr/bin/env python
# coding: utf-8

# In[73]:


from sklearn.datasets import load_breast_cancer
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.spatial import cKDTree

def mae(input):
    x_train = X_train[:,input]
    x_test = X_test[:,input]
    uni_knr=KNeighborsRegressor(weights='uniform')   #初始化平均回归的KNN回归器
    uni_knr.fit(x_train,y_train)
    uni_knr_y_predict=uni_knr.predict(x_test)
    mae = mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))
    # dis_knr=KNeighborsRegressor(weights='distance')   #初始化距离加权回归的KNN回归器
    # dis_knr.fit(X_train,y_train)
    # dis_knr_y_predict=dis_knr.predict(X_test)

    return mae

def GammaTest(X,Y,input):
    p = 10
    # xnik = np.array([])
    M = X.shape[0]
    ynik = np.array([])
    delta = np.zeros(p)
    gamma = np.zeros(p)
    X = np.asarray(X.iloc[:,input])
    tree = cKDTree(X)
    dist, ind = tree.query(X,k=p+1,p=2) #ind.shape()=M*p
    ind = np.delete(ind,0,axis=1)
    dist = np.delete(dist,0,axis=1)
    # xnik = np.zeros((M*k,p))
    ynik = np.zeros((M,p))
    # xnik = x[ind]
    for row in range(M):
        for col in range(p):
            ynik[row,col] = Y[ind[row][col]]
        # for col in range(1,p):
        #     tmp = x[ind[:,col],:] - x[col,:].reshape(1,-1)
            # xnik = np.hstack([ xnik,x[ind[:,col]] ])
    #     # ynik = np.vstack([ynik,(y[ind[:,col]]- y[col])])
        # ynik = np.vstack([ynik,(y[ind[col,:]])])
    for row in range(M):
        ynik[row,:] -=Y[row]

    delta =np.sum(dist**2,axis=0)/M
    gamma = np.sum(ynik**2,axis=0)/(2*M)
    # delta =np.mean(dist**2,axis=0)
    # gamma = np.mean(ynik**2,axis=0)
    # assert dist**2 == xnik**2
    AandT = np.polyfit(delta,gamma,1)
    return AandT[1]

def get_reward(X,Y,features,K):
    if len(features)==0:
        return 0
    else:
        R =  0.01/GammaTest(X,Y,features)
        # R =  0.01/mae(features)
        tot_f = len(features)
        if tot_f>K:
            R = R*K/tot_f
    return R

def online_greedyAction(Q_value,ActionTable,agent):
    actionIndex = np.argmax(Q_value[agent])   # index of maximum Q-value
    return actionIndex
    #action = ActionTable[agent][actionIndex]
    #return action

# Random exploration strategy
def online_randomAction(Q_value,ActionTable,agent):
    actionIndex = random.randrange(0,2)
    return actionIndex
    #action = ActionTable[agent][actionIndex]
    #return action

# Epsilon-greedy exploration strategy
def offline_eGreedy(Q_value,ActionTable,agent,epsilon):
    # Choose action
    temp = np.argmax(Q_value[agent])   # index of maximum Q-value

    rand = random.uniform(0,1)
    if rand > epsilon:
        # Choose greedy action
        counterfactualIndex = temp
    else:
        # Choose random action
        counterfactualIndex = random.randrange(0,2)
        while counterfactualIndex == temp:
            counterfactualIndex = random.randrange(0,2)

    return counterfactualIndex
    #counterfactual = ActionTable[agent][counterfactualIndex]
    #return counterfactual

# Random exploration strategy
def offline_randomAction(Q_value,ActionTable,agent):
    counterfactualIndex = random.randrange(0,2)
    return counterfactualIndex
    #counterfactual = ActionTable[agent][counterfactualIndex]
    #return counterfactual

epsilon = 0.15
alpha = 0.2
epsilon_decay_rate = 0.9995
alpha_decay_rate = 0.9995
all_rewards = []
num_episodes = 100
reward_store={}
flag_CLEAN=1

# data = pd.read_csv('abalone.csv')
data = pd.read_csv('puma32h.csv',header = 0)
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
num_agents = X.shape[1]
K = num_agents
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(np.asarray(y_train).reshape(-1, 1))
y_test=ss_y.transform(np.asarray(y_test).reshape(-1, 1))

Q_value = -np.ones([num_agents, 2], dtype = int)
ActionTable = np.ones([num_agents, 2], dtype = int)
features_store = {}
ActionTable[:,0] = 0
for i in range(num_agents+1):
    reward_store[i]=0
    features_store[i]=0
actions = [0] *num_agents

for episode in range(num_episodes):
    for agent in range(num_agents):
        if flag_CLEAN ==1:
            action = online_greedyAction(Q_value,ActionTable,agent)
        else:
            action = online_randomAction(Q_value,ActionTable,agent)
        actions[agent]=action
    features = []
    for i, act in enumerate(actions):
        if act == 1:
            features.append(i)
    R = get_reward(X,Y,features,K)
    if reward_store[len(features)] < R:
        features_store[len(features)] = features
    reward_store[len(features)]=max(reward_store[len(features)],R)
    for agent in range(num_agents):
        if flag_CLEAN ==1:
            counterfactual = offline_eGreedy(Q_value,ActionTable,agent,epsilon)
            actions[counterfactual]=action
            features = []
            for i, act in enumerate(actions):
                if act == 1:
                    features.append(i)
            C_agent = get_reward(X,Y,features,K) - R
            Q_value[agent][counterfactual] = Q_value[agent][counterfactual]+ alpha * (C_agent - Q_value[agent][counterfactual])
        else:
            action = online_randomAction(Q_value,ActionTable,agent)
            Q_value[agent][action] = Q_value[agent][action]+ alpha * (R - Q_value[agent][action])
    alpha = alpha * alpha_decay_rate
    epsilon = epsilon * epsilon_decay_rate

uni_knr=KNeighborsRegressor(weights='uniform')   #初始化平均回归的KNN回归器
uni_knr.fit(X_train,y_train)
uni_knr_y_predict=uni_knr.predict(X_test)
mae_allset = mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict))
reward_store[7] = 0.01/mae_allset
print(reward_store)
print(features_store)
# actions = [0] *num_agents
# for episode in range(num_episodes):
#         for agent in range(num_agents):
#                 rand_number = random.uniform(0,1)
#                 if rand_number>epsilon:
#                     #actions[agent]  = Q_values[agent].index(max(Q_values[agent]))
#                     actions[agent] = np.argmax(Q_values[agent])
#                 else:
#                     actions[agent] = random.choice([0,1])
#         features = []
#         for i, act in enumerate(actions):
#             if act == 1:
#                 features.append(i)
#         #print(features)
#         R = get_reward(features)
#         reward_store[len(features)]=max(reward_store[len(features)],R)
#         #print(R)
#         all_rewards.append(R)
#         for agent in range (num_agents):
#                 actions[agent] = 1-actions[agent]
#                 features = []
#                 for i, act in enumerate(actions):
#                     if act == 1:
#                         features.append(i)
#                 C_agent = get_reward(features) - R
#                 Q_values[agent][actions[agent]] = Q_values[agent][actions[agent]] + alpha*(C_agent - Q_values[agent][actions[agent]])
#         alpha = alpha * alpha_decay_rate
#         epsilon = epsilon * epsilon_decay_rate







