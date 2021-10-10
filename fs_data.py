from swarm import Swarm
from fs_problem import FsProblem
import pandas as pd
import os, re, time, sys
from rl import QLearning
from solution import Solution
import xlsxwriter
from sklearn import ensemble
from sklearn.metrics import accuracy_score,auc,roc_curve,classification_report,explained_variance_score, mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,KFold
import numpy as np

class FSData():

    def __init__(self,typeOfAlgo,location,nbr_exec, method, test_param, param, val, regressor, alpha=None,gamma=None,epsilon=None):

        self.typeOfAlgo = typeOfAlgo
        self.location = location
        self.nb_exec = nbr_exec
        self.dataset_name = re.search('[A-Za-z\-]*.csv',self.location)[0].split('.')[0]
        self.df = pd.read_csv(self.location,header=0)
        self.ql = QLearning(len(self.df.columns)-1,Solution.attributs_to_flip(len(self.df.columns)-1),alpha,gamma,epsilon)
        self.fsd = FsProblem(self.typeOfAlgo,self.df,self.ql)
        self.regressor_name = str(type(self.fsd.regressor)).strip('< > \' class ').split('.')[3]
        path = './results/parameters/'+method+'/'+test_param+'/'+param+'/'+val+'/'+regressor+'/'+ self.dataset_name
        if not os.path.exists(path):
          os.makedirs(path + '/logs/')
          os.makedirs(path + '/sheets/')
        self.instance_name = self.dataset_name + '_' +  str(time.strftime("%m-%d-%Y_%H-%M-%S_", time.localtime()) + self.regressor_name)
        # log_filename = str(path + '/logs/'+ self.instance_name)
        # if not os.path.exists(path):
        #   os.makedirs(path)
        # log_file = open(log_filename + '.txt','w+')
        # sys.stdout = log_file
        
        # print("[START] Dataset " + self.dataset_name + " description \n")
        # print("Shape : " + str(self.df.shape) + "\n")
        # print(self.df.describe())
        # print("\n[END] Dataset " + self.dataset_name + " description\n")
        # print("[START] Ressources specifications\n")
        # #os.exec('cat /proc/cpuinfo') # Think of changing this when switching between Windows & Linux
        # print("[END] Ressources specifications\n")

        
        # sheet_filename = str(path + '/sheets/'+ self.instance_name )
        # self.workbook = xlsxwriter.Workbook(sheet_filename + '.xlsx')
        
        # self.worksheet = self.workbook.add_worksheet(self.regressor_name)
        # self.worksheet.write(0,0,"Iteration")
        # self.worksheet.write(0,1,"MSE")
        # self.worksheet.write(0,2,"N_Features")
        # self.worksheet.write(0,3,"Time")
        # self.worksheet.write(0,4,"Top_10%_features")
        # self.worksheet.write(0,5,"Size_sol_space")
    
    def run(self,flip,max_chance,bees_number,maxIterations,locIterations):
        total_time = 0
        for itr in range(1,self.nb_exec+1):
          print ("Execution {0}".format(str(itr)))
          self.fsd = FsProblem(self.typeOfAlgo,self.df,self.ql)
          swarm = Swarm(self.fsd,flip,max_chance,bees_number,maxIterations,locIterations)
          t1 = time.time()
          best = swarm.bso(self.typeOfAlgo,flip)
          t2 = time.time()
          total_time += t2-t1
          print("Time elapsed for execution {0} : {1:.2f} s\n".format(itr,t2-t1))
          print("{0}".format(str([j[0] for j in [i for i in swarm.best_features()]])))
          print(len(Solution.solutions))
          # self.worksheet.write(itr, 0, itr)
          # self.worksheet.write(itr, 1, "{0:.2f}".format(best[0]))
          # self.worksheet.write(itr, 2, best[1])
          # self.worksheet.write(itr, 3, "{0:.3f}".format(t2-t1))
          # self.worksheet.write(itr, 4, "{0}".format(str([j[0] for j in [i for i in swarm.best_features()]])))
          # self.worksheet.write(itr, 5, len(Solution.solutions))
          
        print ("Total execution time of {0} executions \nfor dataset \"{1}\" is {2:.2f} s".format(self.nb_exec,self.dataset_name,total_time))
        #self.workbook.close()
        bestSol = list(np.array(best[2]).nonzero()[0])
        best_subset = [j[0] for j in [i for i in swarm.best_features()]]
        cv = 5
        kfold = KFold(n_splits=cv, random_state=123, shuffle=True)
        tmp_list = []
        tmp_score = 0.0
        tmp_list1 = []
        tmp_score1 = 0.0
        start = time.time()
        params = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.2, 'loss': 'ls'}
        score = 0
        X = self.df.iloc[:, 0:len(self.df.columns)-1]
        Y = self.df.iloc[:,-1]
        for train, test in kfold.split(X, Y):
            X_train = X.loc[train,:]
            X_test = X.loc[test,:]
            y_train = Y[train]
            y_test = Y[test]
    
            model_gbt = ensemble.GradientBoostingRegressor(**params)
            model_gbt.fit(X_train.iloc[:,best_subset], Y[train])
            y_pre = model_gbt.predict(X_test.iloc[:,best_subset])
            model_gbt1 = ensemble.GradientBoostingRegressor(**params)
            model_gbt1.fit(X_train.iloc[:,bestSol], Y[train])
            y_pre1 = model_gbt1.predict(X_test.iloc[:,bestSol])
            # model_gbt = SVR(kernel='rbf',C=80, gamma = 1/len(best_subset)) # 建立支持向量机回归模型对象, C=1e3, gamma = 0.
            # model_gbt.fit(X_train[best_subset],y_train)
            # y_pre = model_gbt.predict(X_test[best_subset])
            
            #R-Squared 越大，表示模型拟合效果越好。
            model_metrics_name = [explained_variance_score, mean_absolute_error,mean_squared_error,r2_score]
            for mdl in model_metrics_name:
                tmp_score = mdl(y_test, y_pre)
                tmp_list.append(tmp_score)
                tmp_score1 = mdl(y_test, y_pre1)
                tmp_list1.append(tmp_score1)
        tmp_list = np.array(tmp_list).reshape(cv,-1).sum(axis = 0)/cv
        tmp_list1 = np.array(tmp_list1).reshape(cv,-1).sum(axis = 0)/cv
        df = pd.DataFrame(tmp_list,columns = ['model_gbt'], index = ['ev','mae','mse','r2'])
        print(df)
        df1 = pd.DataFrame(tmp_list1,columns = ['model_gbt1'], index = ['ev','mae','mse','r2'])
        print(df1)
