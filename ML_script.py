# -*- coding: utf-8 -*-
# @Time    : 2024/11/26 23:53
# @Author  : aries.yu
from sklearn.utils import shuffle
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def rbf_NuSVR(X_train, X_test, y_train, y_test, loop):
    nusvr = NuSVR()
    parameters = [
        {'C': [0.1, 1e0, 1e1, 1e2,1e3], 'gamma': [0.001,0.01, 0.1, 0.5],
         'kernel': ['rbf'],}]
    #clf1 = GridSearchCV(nusvr, parameters, scoring='neg_mean_squared_error', cv=5)
    clf1 = GridSearchCV(nusvr, parameters, scoring='neg_mean_absolute_error', cv=5)
    clf1.fit(X_train, y_train)
    param = clf1.best_params_
    print(param)
    svr1 = NuSVR(C=param.get("C"), gamma=param.get("gamma"), kernel=param.get("kernel"),)
    svr1.fit(X_train,y_train)
    
    print("*****************RBF:")
    print(dir(svr1))
    
    print("clsss_weight:")
    print(svr1.class_weight_)
    print("\n")
    print("coef:")
    print(svr1.coef_)
    print("\n")
    print("dual_coef:")
    print(svr1.dual_coef_)
    print("\n")
    print("probA:")
    print(svr1.probA_)
    print("\n")
    print("probB:")
    print(svr1.probB_)
    print("\n")
    print("shape_fit:")
    print(svr1.shape_fit_)
    print("\n")
    
    y_pred=svr1.predict(X_test)
    y_train_pred=svr1.predict(X_train)

    print(y_train)
    print(y_train_pred)
    #y_test
    #y_pred
    dftrainname="%d_rbf_train.csv"%(loop)
    dftestname="%d_rbf_test.csv"%(loop)

    result_train_df=pd.DataFrame()
    result_train_df["y_train"]=y_train
    result_train_df["y_train_pred"]=list(y_train_pred.reshape(1,-1)[0])
    
    result_test_df=pd.DataFrame()
    result_test_df["y_test"]=y_test
    result_test_df["y_pred"]=list(y_pred.reshape(1,-1)[0])

    result_train_df.to_csv("./%s"%(dftrainname),index=False)
    result_test_df.to_csv("./%s"%(dftestname),index=False)
    print(result_train_df)

    train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_score = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2=r2_score(y_test, y_pred)
    train_r2=r2_score(y_train, y_train_pred)
    train_mae=mean_absolute_error(y_train, y_train_pred)
    test_mae=mean_absolute_error(y_test, y_pred)
    
    return train_r2 ,test_r2, train_score, test_score, train_mae, test_mae

def linear_nuSVR(X_train, X_test, y_train, y_test):
    nusvr = NuSVR()
    parameters = [
        {'C': [0.1, 1e0, 1e1, 1e2,1e3], 'gamma': [0.001,0.01, 0.1, 0.5],
         'kernel': ['linear'],}]
    clf1 = GridSearchCV(nusvr, parameters, scoring='neg_mean_squared_error', cv=5)
    clf1.fit(X_train, y_train)
    param = clf1.best_params_
    svr1 = NuSVR(C=param.get("C"), gamma=param.get("gamma"), kernel=param.get("kernel"),)#nu=param.get("nu")

    svr1.fit(X_train,y_train)
    y_pred=svr1.predict(X_test)
    y_train_pred=svr1.predict(X_train)
    train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_score = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2=r2_score(y_test, y_pred)
    train_r2=r2_score(y_train, y_train_pred)
    train_mae=mean_absolute_error(y_train, y_train_pred)
    test_mae=mean_absolute_error(y_test, y_pred)

    return train_r2, test_r2, train_score, test_score, train_mae, test_mae 

def rf(X_train, X_test, y_train, y_test, fetures, loop):
    rf=RandomForestRegressor()

    parameters = [
        {'n_estimators': [20,50,70], 'max_depth':[3,4,5,7,10],
         "min_samples_split":[2,4,6,10]}]
    clf1 = GridSearchCV(rf, parameters, scoring='neg_mean_squared_error', cv=5)
    clf1.fit(X_train, y_train)
    param = clf1.best_params_
    svr1 =RandomForestRegressor(n_estimators=param.get("n_estimators"), max_depth=param.get("max_depth"),
                                min_samples_split=param.get("min_samples_split"))
    svr1.fit(X_train,y_train)
   
    features=np.array([fetures]).reshape(-1,1)
    rf_feature_importance=svr1.feature_importances_
    print(rf_feature_importance)
    rf_feature_importance=np.array([rf_feature_importance]).reshape(-1,1)
    features_and_importance=np.concatenate((features, rf_feature_importance), axis=1)
    
    print(features_and_importance)

    feature_and_importance_sort=features_and_importance[np.lexsort(features_and_importance.T)]
    feature_and_importance_sort=feature_and_importance_sort[::-1]
    print(feature_and_importance_sort)
    featuresimportance=pd.DataFrame(feature_and_importance_sort, columns=["feature", "importance"])
    featurefilename="%d_rf_feature_importance.csv"%(loop)
    featuresimportance.to_csv("./%s"%(featurefilename), index=False)

    print(dir(svr1))
    print("***************estimators:")
    print(svr1.estimators_)
    print("\n")
    print("***************feature_importances:")
    print(svr1.feature_importances_)
    print("\n")
    print("***************n_features:")
    print(svr1.n_features_)
    print("\n")
    print("***************n_outputs:")
    print(svr1.n_outputs_)
    print("\n")

    y_pred=svr1.predict(X_test)
    y_train_pred=svr1.predict(X_train)

    dftrainname="%d_rf_train.csv"%(loop)
    dftestname="%d_rf_test.csv"%(loop)

    result_train_df=pd.DataFrame()
    result_train_df["y_train"]=y_train
    result_train_df["y_train_pred"]=list(y_train_pred.reshape(1,-1)[0])
    
    result_test_df=pd.DataFrame()
    result_test_df["y_test"]=y_test
    result_test_df["y_pred"]=list(y_pred.reshape(1,-1)[0])
    result_train_df.to_csv("./%s"%(dftrainname),index=False)
    result_test_df.to_csv("./%s"%(dftestname),index=False)
 
    train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_score = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2=r2_score(y_test, y_pred)
    train_r2=r2_score(y_train, y_train_pred)

    train_mae=mean_absolute_error(y_train, y_train_pred)
    test_mae=mean_absolute_error(y_test, y_pred)

    return train_r2, test_r2, train_score, test_score, train_mae, test_mae 

def loops(num=20):    
    
    rbf_array=[]
    linear_array=[]
    rf_array=[]
    loopindex=[]

    for loop in range(1,num+1):
        print("loop %d... " %(loop))
        data=pd.read_csv(r"./Sil-CE.csv")
        data = shuffle(data)

        df_corr=data.corr()
        df_corr.to_csv(r"df_corr.csv")
#        data.drop(["LMBL","LMBLDELTA"],axis=1,inplace=True)
        y=data["Energy"]
        X=data.drop("Energy",axis=1)
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
        std=MinMaxScaler()
        X_train=std.fit_transform(X_train)
        X_test=std.transform(X_test)
        
        features = list(X.columns)
        #print(features)
        loopindex.append(int(loop))
        #rbf_record=[loop]
        
        rbf_record=[]
        rbflisttmp=list(rbf_NuSVR(X_train, X_test, y_train, y_test, loop))
        rbf_record.extend(rbflisttmp)
        rbf_array.append(rbf_record)
       
        #"""
        rf_record=[]
        rflisttmp=list(rf(X_train, X_test, y_train, y_test, features, loop))
        print(rf_feature_importance)

        rf_record.extend(rflisttmp)
        rf_array.append(rf_record)
        #"""
    #print("loop_rbf ,rbf_train_r2, rbf_test_r2, rbf_train_rmse, rbf_test_rmse, rbf_train_mae, rbf_test_mae, loop_rf, rf_train_r2, rf_test_r2, rf_train_rmse, rf_test_rmse, rf_train_mae, rf_test_mae")
    
    rbf_array_result=np.array(rbf_array)
    rbf_array_mean=["mean_rbf"]
    rbf_array_mean.extend(list(np.mean(rbf_array_result, axis=0)))
    rbf_array_mean.extend(["NAN", "NAN"])
    rbf_array_std=["std_rbf"]
    rbf_array_std.extend(list(np.std(rbf_array_result, axis=0)))
    rbf_array_std.extend(["NAN", "NAN"])
    rbf_array_result_6=rbf_array_result[:,5]
    rbf_test_mae_mean=np.mean(rbf_array_result, axis=0)[5]
    loopindex_array=np.array(loopindex).reshape(-1,1)
    loopindex_array.dtype=int
    error_rbf=rbf_array_result_6.reshape(-1,1) - rbf_test_mae_mean.reshape(-1,1)
    error_rbf_absolute=np.absolute(error_rbf)
    rbf_data_all=np.concatenate((loopindex_array,rbf_array_result, error_rbf_absolute), axis=1)
    rbf_data_all_sort=np.concatenate((rbf_data_all[np.lexsort(rbf_data_all.T)],loopindex_array), axis=1)
    title_rbf=["rbf_loop" ,"rbf_train_r2", "rbf_test_r2", "rbf_train_rmse", "rbf_test_rmse", "rbf_train_mae", "rbf_test_mae", "rbf_mae_error", "rbf_mae_error_index"]
    print(title_rbf)
    print(rbf_data_all_sort)
    rbf_data_mean_std=np.concatenate((np.array([title_rbf]),rbf_data_all_sort, np.array([rbf_array_mean]), np.array([rbf_array_std])), axis=0)
    print(rbf_data_mean_std)
    print(rbf_test_mae_mean)
    
    rf_array_result=np.array(rf_array)
    rf_array_mean=["mean_rf"]
    rf_array_mean.extend(list(np.mean(rf_array_result, axis=0)))
    rf_array_mean.extend(["NAN", "NAN"])
    rf_array_std=["std_rf"]
    rf_array_std.extend(list(np.std(rf_array_result, axis=0)))
    rf_array_std.extend(["NAN", "NAN"])
    rf_array_result_6=rf_array_result[:,5]
    rf_test_mae_mean=np.mean(rf_array_result, axis=0)[5]
    loopindex_array=np.array(loopindex).reshape(-1,1)
    loopindex_array.dtype=int
    error_rf=rf_array_result_6.reshape(-1,1) - rf_test_mae_mean.reshape(-1,1)
    error_rf_absolute=np.absolute(error_rf)
    rf_data_all=np.concatenate((loopindex_array,rf_array_result, error_rf_absolute), axis=1)
    rf_data_all_sort=np.concatenate((rf_data_all[np.lexsort(rf_data_all.T)],loopindex_array), axis=1)
    title_rf=["rf_loop" ,"rf_train_r2", "rf_test_r2", "rf_train_rmse", "rf_test_rmse", "rf_train_mae", "rf_test_mae", "rf_mae_error", "rf_mae_error_index"]
    print(title_rf)
    rf_data_mean_std=np.concatenate((np.array([title_rf]) ,rf_data_all_sort, np.array([rf_array_mean]), np.array([rf_array_std])), axis=0)
    print(rf_data_mean_std)

    rbf_rf=np.concatenate((rbf_data_mean_std, rf_data_mean_std), axis=0)
    print(rbf_rf)
    
    scoreing=pd.DataFrame(rbf_rf)
    columns=["loop" ,"train_r2", "test_r2", "train_rmse", "test_rmse", "train_mae", "test_mae", "mae_error", "mae_error_index"]
    scoreing=pd.DataFrame(rbf_rf, columns=columns)
    scoreing.to_csv(r"./scores.csv", index=False)

    #alldata=np.concatenate((rbf_data, rf_data), axis=1)
    #alldata=rbf_data
    np.set_printoptions(precision=5)
    #print(alldata)
    #linear_array=np.array(linear_array)
    #print(linear_array)

if __name__ == '__main__':
    
    #loopnum=2
    loops(20)
    
# result_lst=[train_socre,test_socre,train_r2,test_r2,train_mae,test_mae]
# result_all_df=pd.DataFrame([result_lst],columns=["train_socre","test_socre","train_r2","test_r2","train_mae","test_mae"])
# result_all_df.to_csv(r"rbf_result_all_df.csv",index=False)
#
# result_train_df=pd.DataFrame()
# result_test_df=pd.DataFrame()
# result_train_df["y_train"]=y_train
# result_train_df["y_train_pred"]=list(y_train_pred.reshape(1,-1)[0])
# result_test_df["y_test"]=y_test
# result_test_df["y_pred"]=list(y_pred.reshape(1,-1)[0])
# result_train_df.to_csv(r"rbf_train_pred_all_features.csv",index=False)
# result_test_df.to_csv(r"rbf_test_pred_all_features.csv",index=False)



