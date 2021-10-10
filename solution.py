import copy, time, operator

class Solution:

    solutions = {} 
    best_sol = None
    tot_eval_time = 0
    sorting_time = 0

    def __init__(self,data,state):
        self.data = data
        self.state = state
        self.mse = 100.0
        self.solutions[Solution.str_sol(self.state)] = self.mse

    def get_mse(self,state):
        if (Solution.str_sol(state) in Solution.solutions):
            if (Solution.solutions[Solution.str_sol(state)] == 100.0) :
                self.set_mse(state)
        else:
          self.set_mse(state)
        return Solution.solutions[Solution.str_sol(state)]

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_mse(self,state): 
        t1 = time.time()
        Solution.solutions[Solution.str_sol(state)] = self.data.evaluate(state)
        self.mse = Solution.solutions[Solution.str_sol(state)]
        t2 = time.time()
        Solution.tot_eval_time += t2-t1
        if (Solution.best_sol == None) or (Solution.best_sol.get_mse(Solution.best_sol.get_state()) < self.mse):
            Solution.best_sol = self
    
    def set_state(self,state): 
        self.state = copy.deepcopy(state)

    @staticmethod
    def get_best_sol():
      # This part has been changed by a variable "best_sol", because sorting was costing some execution time
        """t1 = time.time()
        sorted_sols = sorted(Solution.solutions.items(), key=operator.itemgetter(1), reverse=True)
        t2 = time.time()
        #print("Best sol after sort : {0}".format(sorted_sols[0][1]))
        Solution.sorting_time += t2-t1
        return sorted_sols[0][0] ,sorted_sols[0][1]"""
        if Solution.best_sol == None :
          best_state = Solution.str_sol(list(Solution.solutions.items())[0][0])
          best_mse = list(Solution.solutions.items())[0][1]
        else:
          best_state = Solution.best_sol.get_state()
          best_mse = Solution.best_sol.get_mse(best_state)
        return Solution.str_sol(best_state), best_mse

    @staticmethod
    def get_indexes(mlist):
        ilist = []
        for i in range(len(mlist)):
          if mlist[i] == 1:
            ilist.append(i)
        return ilist

    @staticmethod
    def str_sol(mlist):
        result = ''
        for element in mlist:
            result += str(element)
        return result

    @staticmethod
    def sol_to_list(solution):
        sol_list=[i for i, n in enumerate(solution) if n == 1]
        return sol_list

    @staticmethod
    def list_sol(key):
        mlist = [ int(i) for i in key ]
        return mlist

    @staticmethod
    def nbrUn(state):
        return len([i for i, n in enumerate(state) if n == 1])

    @staticmethod
    def attributs_to_flip(nb_att):
        return list(range(nb_att))

    @staticmethod
    def xor(x, y):
        return '{1:0{0}b}'.format(len(x), int(Solution.str_sol(x), 2) ^ int(Solution.str_sol(y), 2))

    @staticmethod
    def get_avg_time():
        return Solution.tot_eval_time/len(Solution.solutions)