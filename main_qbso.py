from fs_data import FSData
if __name__=="__main__":

    # RL 
    alhpa = 0.1
    gamma = 0.9
    epsilon = 0.1

    # BSO

    flip = 5
    max_chance = 3
    bees_number = 10
    maxIterations = 10
    locIterations = 10

    # Test type

    typeOfAlgo = 1 # 1 for ql localsearch 0 for normal localsearch
    nbr_exec = 1
    dataid = 6
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
    method = "qbso_simple"
    test_param = "rl"
    param = "gamma"
    val = str(locals()[param])
    regressor = "knn"
    instance = FSData(typeOfAlgo,location,nbr_exec,method,test_param,param,val,regressor,alhpa,gamma,epsilon)
    instance.run(flip,max_chance,bees_number,maxIterations,locIterations)