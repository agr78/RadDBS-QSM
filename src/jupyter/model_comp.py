
# Import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.pipeline as spl
import sklearn.kernel_ridge as skr
import sklearn.model_selection as sms
import sklearn.linear_model as slm
import sklearn.preprocessing as skp
import sklearn.neural_network as snn
import sklearn.metrics as sme
import sklearn.decomposition as sdc
import sklearn.cross_decomposition as skd
import sklearn.feature_selection as skf
import sklearn.ensemble as ske
from sklearnex import patch_sklearn, config_context
from sklearn.cluster import DBSCAN
import numpy as np
from IPython.display import HTML
import util

patch_sklearn()

# Get case IDs
case_list = open('/home/ali/RadDBS-QSM/data/docs/cases_90','r')
lines = case_list.read()
lists = np.loadtxt(case_list.name,comments="#", delimiter=",",unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

# Load scores
file_dir = '/home/ali/RadDBS-QSM/data/docs/QSM anonymus- 6.22.2023-1528.csv'
motor_df = util.filter_scores(file_dir,'pre-dbs updrs','stim','CORNELL ID')
# Find cases with all required scores
subs,pre_imp,post_imp,pre_updrs_off = util.get_full_cases(motor_df,
                                                          'CORNELL ID',
                                                          'OFF (pre-dbs updrs)',
                                                          'ON (pre-dbs updrs)',
                                                          'OFF meds ON stim 6mo')
# Load extracted features
npy_dir = '/home/ali/RadDBS-QSM/data/npy/'
phi_dir = '/home/ali/RadDBS-QSM/data/phi/'
roi_path = '/data/Ali/atlas/mcgill_pd_atlas/PD25-subcortical-labels.csv'
n_rois = 6
Phi_all, X_all, R_all, K_all, ID_all = util.load_featstruct(phi_dir,npy_dir+'X/',npy_dir+'R/',npy_dir+'K/',n_rois,1595)
ids = np.asarray(ID_all).astype(int)
# Find overlap between scored subjects and feature extraction cases
c_cases = np.intersect1d(np.asarray(case_id).astype(int),np.asarray(subs).astype(int))
# Complete case indices with respect to feature matrix
c_cases_idx = np.in1d(ids,c_cases)
X_all_c = X_all[c_cases_idx,:,:]
K_all_c = K_all[c_cases_idx,:,:]
R_all_c = R_all[c_cases_idx,:,:]
# Re-index the scored subjects with respect to complete cases
s_cases_idx = np.in1d(subs,ids[c_cases_idx])
pre_imp = pre_imp[s_cases_idx]
post_imp = post_imp[s_cases_idx]
pre_updrs_off = pre_updrs_off[s_cases_idx]
per_change = post_imp

# Cross validation
cvn = 8
num_splits = 100
for j in np.arange(num_splits):
        
    # Split data
    X_train,X_test,y_train,y_test,train_index,test_index = util.set_split(X_all_c,per_change,1,5/len(X_all_c))

    # Choose scaling
    X0_ss0,scaler_ss,X_test_ss0 = util.model_scale(skp.StandardScaler(),
                                                X_train,train_index,X_test,test_index,pre_updrs_off)
    X0_mm,scaler_mm,X_test_mm = util.model_scale(skp.MinMaxScaler(),
                                                X_train,train_index,X_test,test_index,pre_updrs_off)
    X0_ma,scaler_ma,X_test_ma = util.model_scale(skp.MaxAbsScaler(),
                                                X_train,train_index,X_test,test_index,pre_updrs_off)
    X0_rs,scaler_rs,X_test_rs = util.model_scale(skp.RobustScaler(),
                                                X_train,train_index,X_test,test_index,pre_updrs_off)
    # Feature selection
    sel = skf.SelectKBest(skf.f_regression,k=100)
    X0_ss = sel.fit_transform(X0_ss0,y_train)
    X_test_ss = (sel.transform(X_test_ss0.reshape(X_test_ss0.shape[0],X_test_ss0.shape[1]*X_test_ss0.shape[2]))).reshape((X_test_ss0.shape[0],1,-1))


    scoring = 'r2'
    print(y_test)
    print(y_test.mean())
    print(y_train.mean())

    alphas = np.logspace(-9,-2,10)

    lr = slm.LinearRegression()
    est_lr = lr.fit(X0_ss,y_train)
    results_lr = est_lr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_lr)

    br_grid = {'alpha_1': alphas[-5:-4], 'alpha_2': alphas[-5:-4]}

    best_params = util.gridsearch_pickparams(slm.BayesianRidge(),cvn,
                                            br_grid,scaler_ss,X_train,
                                            train_index,X_test,test_index,pre_updrs_off,y_train.ravel(),scoring,8)
    br = slm.BayesianRidge(alpha_1=best_params['alpha_1'],alpha_2=best_params['alpha_2'])
    br.fit(X0_ss, y_train)
    results_br = np.asarray(br.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()
    print(results_br)

    mlp_grid = {'hidden_layer_sizes': [(X_train.shape[1],X_train.shape[2])],
            'activation': ['relu'],
            'alpha': alphas,
            'epsilon': [1e0],
            'solver': ['adam'],
            'max_iter':[5000]}

    best_params = util.gridsearch_pickparams(snn.MLPRegressor(),
                                            cvn,mlp_grid,scaler_ss,X_train,
                                            train_index,X_test,test_index,pre_updrs_off,y_train,scoring,-1)

    mlp = snn.MLPRegressor(hidden_layer_sizes=best_params["hidden_layer_sizes"], 
                            activation=best_params["activation"],
                            solver=best_params["solver"],
                            alpha=best_params['alpha'],
                            epsilon=best_params["epsilon"],
                            max_iter=5000, 
                            n_iter_no_change=500, 
                            verbose=True,
                            early_stopping=True,
                            random_state=1,
                            batch_size=len(X0_ss)//cvn)

    mlp.fit(X0_ss,y_train)
    results_mlp = mlp.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))

    print(results_mlp)

    lasso = slm.LassoCV(
        alphas=alphas,
        cv=cvn, 
        verbose=True,
        random_state=1,
        max_iter=100000,
        tol=1e-3,
        n_jobs=-1)

    est_ls = lasso.fit(X0_ss,y_train)
    results_ls = est_ls.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_ls)

    ridge = slm.RidgeCV(
        alphas=alphas,
        scoring=scoring,
        cv=cvn)

    est_rr = ridge.fit(X0_ss,y_train)
    results_rr = est_rr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_rr)

    lars = slm.LarsCV(
        cv=cvn, 
        max_iter=1000,
        max_n_alphas=10000,
        verbose=True,
        normalize=False,
        eps=np.finfo(float).eps,
        n_jobs=-1)

    est_lars = lars.fit(X0_ss,y_train)
    results_lars = est_lars.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_lars)

    krr_grid = {'kernel': ['linear','rbf'],
            'alpha': [alphas]}

    best_params = util.gridsearch_pickparams(skr.KernelRidge(),
                                            cvn,krr_grid,scaler_ss,X_train,train_index,
                                            X_test,test_index,pre_updrs_off,y_train,scoring,-1)
    krr = skr.KernelRidge(kernel=best_params['kernel'],alpha=best_params['alpha'])
    krr.fit(X0_ss, y_train)
    results_krr = krr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_krr)

    gsc = slm.ElasticNetCV(
        alphas=alphas,
        cv=cvn, 
        max_iter=10000,
        verbose=True,
        n_jobs=-1)

    est_en = gsc.fit(X0_ss,y_train)
    results_en = est_en.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))
    print(results_en)

    pls_grid = {'n_components': np.flip(np.arange(5,int(len(X_train)))),
                'scale': [True,False]}

    best_params = util.gridsearch_pickparams(skd.PLSRegression(),cvn,
                                            pls_grid,scaler_ss,X_train,
                                            train_index,X_test,test_index,pre_updrs_off,y_train,scoring,-1)
    pls = skd.PLSRegression(n_components=best_params['n_components'],scale=best_params['scale'],max_iter=10000)
    pls.fit(X0_ss, y_train)
    results_pls = (pls.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()

    print(results_pls)

    pcr = spl.make_pipeline(sdc.PCA(),slm.LinearRegression())
    pcr.fit(X0_ss, y_train)
    results_pcr = np.asarray(pcr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()
    print(results_pcr)

    omp = slm.OrthogonalMatchingPursuitCV(normalize=False,cv=cvn,max_iter=len(X_train)//2)
    omp.fit(X0_ss, y_train)
    results_omp = np.asarray(omp.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()
    print(results_omp)

    rsr = slm.RANSACRegressor(random_state=1,min_samples=len(X0_ss)).fit(X0_ss, y_train)
    results_rsr = rsr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]])).ravel()
    print(results_rsr)


    svr_grid = {'kernel': ['linear','rbf'],
            'epsilon': [1e-1,1.5e-1,2.5e-1]}
    best_params = util.gridsearch_pickparams(svm.SVR(),
                                            cvn,svr_grid,scaler_ss,X_train,
                                            train_index,X_test,test_index,pre_updrs_off,y_train,scoring,-1)
    svr = svm.SVR(kernel=best_params['kernel'],epsilon=best_params['epsilon'])
    svr.fit(X0_ss, y_train)
    results_svr = np.asarray(svr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()
    print(results_svr)

    gbr_grid = {'max_depth':[3,6,9,12,15,20,100]}
    best_params = util.gridsearch_pickparams(ske.GradientBoostingRegressor(random_state=1),cvn,
                                            gbr_grid,scaler_ss,X_train,
                                            train_index,X_test,test_index,pre_updrs_off,y_train,scoring,8)
    gbr = ske.GradientBoostingRegressor(random_state=1,learning_rate=0.01,max_depth=best_params['max_depth'],n_estimators=X0_ss.shape[1])
    gbr.fit(X0_ss, y_train)
    results_gbr = np.asarray(gbr.predict(X_test_ss.reshape([X_test_ss.shape[0],
                                            X_test_ss.shape[1]*X_test_ss.shape[2]]))).ravel()

    Ps = np.vstack((pre_imp[test_index],
                                results_lr.ravel(),
                                results_mlp,
                                results_ls,
                                results_lars,
                                results_en,
                                results_rr.ravel(),
                                results_krr.ravel(),
                                results_pcr,
                                results_pls,
                                results_omp,
                                results_br,
                                results_rsr,
                                results_svr,
                                results_gbr,
                                y_test))
    filename = 'Ps_'+str(j)+'.npy'
    np.save(filename,Ps)


