# -*- coding: utf-8 -*-
"""
Source code for ML coursework
The resulting plots produced by this code can be seen in the report
"""

#IMPORT
if 1: 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    import yellowbrick #Machine learning visualisation library
    import sys
    import copy
    import skopt 
    import scikitplot as skplt
    import itertools 

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn import metrics
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.utils import class_weight
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.neighbors import KernelDensity
    from skopt import BayesSearchCV
    from sklearn import linear_model
    from sklearn import kernel_ridge 
    from sklearn import neighbors
    from sklearn import svm
    from sklearn import tree
    from sklearn import ensemble
    from sklearn import naive_bayes
    from yellowbrick import regressor
    from skopt.space import Real,Categorical,Integer
    from sklearn import preprocessing
    from sklearn import utils
    from matplotlib.ticker import FormatStrFormatter
    from imblearn.under_sampling import RandomUnderSampler


    warning = False
    if warning == False:
         pd.options.mode.chained_assignment = None


#%%
#Auxilliary functions for training and hyperparameter optimisation
if 1: 

    
    
    def classification_evaluation(Y_test,Y_pred):
        Y_test = Y_test.reshape(-1,1); Y_pred = Y_pred.reshape(-1,1)
        accuracy = metrics.accuracy_score(Y_test,Y_pred)
        balanced_accuracy = metrics.balanced_accuracy_score(Y_test,Y_pred)
        f1_score = metrics.f1_score(Y_test,Y_pred,average="weighted") #f1 is geometric average of precision and recall. "weighted" is weighted sum across f1 scores for all classes
        #roc_auc = metrics.roc_auc_score(Y_test,Y_pred,average="weighted",multi_class="ovr")
        
        return accuracy, f1_score, balanced_accuracy
    
        
        
        
    def plot_classification(model_results,title="Classification model performance"):
        
        fig,ax = plt.subplots()
        model_names = list(model_results.keys())
        model_index = np.arange(0,len(model_names))
        
        ACC = []; F1 = []; Bal_ACC = []
        
        for model_name in model_names:
            index = model_names.index(model_name)

            model_ACC = model_results[model_name][0]
            model_F1 = model_results[model_name][1]
            #model_Bacc = model_results[model_name][2]
            ACC.append(model_ACC); F1.append(model_F1); #Bal_ACC.append(model_Bacc)
            
        ax.scatter(model_index,ACC,c="b",label="Accuracy",s=100); ax.scatter(model_index,F1,c="r",label="F1",s=100); #ax.scatter(model_index,F1,c="g",label="Bal_acc",s=100)                 
        
        ax.set_ylabel("Metrics",fontsize=20);                                  
        ax.set_xticks(ticks=np.arange(0,len(model_names)));ax.set_xticklabels(labels=model_names,rotation=305,fontsize=20)
        ax.set_yticklabels(labels=ax.get_yticks(),fontsize=15)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        legend =  ax.legend(fancybox=True,framealpha=1,shadow=True,loc="lower right",fontsize=15,frameon=True)
        legend.get_frame().set_alpha(0.5)
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        #ax.grid()
        fig.suptitle(title,fontweight="bold",fontsize=20)
    
    def hyperparameter_optimisation(model,H_space,X_train,Y_train,cv=5,search_type="Grid",verbose=0,sample_weights=None,facecolor="g",scoring="f1_weighted"):
            
        """
        For given model and H_space, returns optimal hyperparameters for model
        """
        
        if search_type=="Bayes":
            Search = BayesSearchCV(model,H_space,cv=5,verbose=verbose,fit_params={"sample_weight":sample_weights},scoring=scoring)
        elif search_type=="Random":
            Search = RandomizedSearchCV(model,H_space,cv=5,verbose=verbose,fit_params={"sample_weight":sample_weights},scoring=scoring)
        else: 
            Search = GridSearchCV(model,H_space,cv=5,verbose=verbose,fit_params={"sample_weight":sample_weights},scoring=scoring)
        
            
        #if sample_weights.all() == None:
        #    sample_weights = np.ones(len(X_train))
            
        Search.fit(X_train,Y_train)
        
        
        score = Search.best_score_
        params = Search.best_params_
        
        return score,params
    
            
    def plot_dist(arr,split=1):
        
        arr = np.round(np.array(arr))
        bins = np.arange(10,80,split)
        
        fig,ax = plt.subplots(figsize=(6,5))
        ax.hist(arr,bins)
        

        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2') 
        

    from yellowbrick.regressor import ResidualsPlot
    
    def plot_learning_curve(model_test,X_train,Y_train,title="Learning Curve"):
        skplt.estimators.plot_learning_curve(model_test, X_train, Y_train,title=title)
    
    
    
    def plot_gender_confusion_matrix(Y_test,Y_pred,name="Confusion matrix"):
        Y_test_copy = copy.deepcopy(Y_test); Y_pred_copy = copy.deepcopy(Y_pred)
        for arr in (Y_test_copy,Y_pred_copy):
            arr[arr<0]=-1; arr[arr>=0]=1
        
        fig,ax = plt.subplots()
        
        skplt.metrics.plot_confusion_matrix(Y_test_copy,Y_pred_copy,
                                            title=name,
                                            cmap="Purples",
                                            ax=ax)
        plt.show()
        
        return Y_test_copy, Y_pred_copy
    
       
        
    def plot_residuals(model_test,train,test):
        fig = plt.figure(figsize=(9,6))
        viz = ResidualsPlot(model_test,
                            train_color="dodgerblue",
                            test_color="green",
                            fig=fig
                            )
        
        viz.fit(train[0],train[1])
        viz.score(test[0],test[1])
        viz.show()
            
        
    def plot_targetdist(Y_train,Y_test=None):
        
        viz = yellowbrick.target.ClassBalance(labels = np.unique(labels))
        if Y_test==None:
            viz.fit(Y_train)

        else:
            assert(type(Y_test)==np.ndarray)
            viz.fit(Y_train,Y_test)
            
        viz.ax.set_xlabel("Classes",fontsize=20); viz.ax.set_ylabel("Frequency",fontsize=20)
        viz.ax.set_xticks(np.arange(0,len(np.unique(labels))),np.unique(labels),rotation=315,fontsize=15); viz.ax.set_yticks(viz.ax.get_yticks(),fontsize=50)

        viz.ax.patch.set_edgecolor('black')  
        viz.ax.patch.set_linewidth('2')
        
        
    def cross_val_k(model,X_train,Y_train,score="accuracy_score"):
        
        folds = np.arange(5,100,10)
        CV_score = []
        
        for fold in folds:
            
            cvscore = sklearn.model_selection.cross_val_score(model,X_train,Y_train,cv=fold)
            cv_score_avg = np.mean(cvscore)
            CV_score.append(cv_score_avg)
            
        fig,ax = plt.subplots()
        fig.suptitle("Cross val "+score+" vs fold")
        ax.plot(folds,CV_score)
        plt.show()
        
        

            
            
        

    
#%% #Data cleaning and transformation



##DATA
data_path = r"C:\Users\iyngk\OneDrive\Public\Documents\Academic\Year 2\AI\Machine Learning\Coursework\HSQ\HSQ\data.csv"
DATA = pd.read_csv(data_path)

if 1:
    
    #fla
    normalisation = False
    w_accuracy = False
    gender_encoding = False
    PCA_features = False
    Class_balancing = False
    polynomial_features = False
    train_weight_calculation = True
    encoding = "integer"

    ##INITIAL DATA CLEANING
    if 1: 
        #Removing 0's in gender column
        DATA = DATA[DATA.gender!=0]
        DATA = DATA[DATA.age<120]

        DATA = DATA[DATA.gender !=3] #Just 8 of other, deemed
        gender = DATA["gender"]

        
        QUESTIONS = DATA.iloc[:,0:32]
        
        #Replace -1 answers with column mode
        QUESTIONS = QUESTIONS.replace(-1,np.nan)
        for Q in QUESTIONS.columns: 
            QUESTIONS[Q].fillna(QUESTIONS[Q].mode()[0], inplace=True)
            
        affiliative = round(((6-DATA["Q1"])+DATA["Q5"]+(6-DATA["Q9"])+DATA["Q13"]+\
                             (6-DATA["Q17"])+DATA["Q21"]+(6-DATA["Q25"])+(6-DATA["Q29"]))/8,1)
    
        selfenhancing = round((DATA["Q2"]+DATA["Q6"]+DATA["Q10"]+DATA["Q14"]+DATA["Q18"]\
                             +DATA["Q22"]+DATA["Q26"]+DATA["Q30"])/8,1)
        
        aggressive = round((DATA["Q3"]+DATA["Q7"]+DATA["Q11"]+DATA["Q15"]+DATA["Q19"]\
                          +DATA["Q23"]+DATA["Q27"]+DATA["Q31"])/8,1)
        
        selfdefeating = round((DATA["Q4"]+DATA["Q8"]+DATA["Q12"]+DATA["Q16"]+DATA["Q20"]\
                          +DATA["Q24"]+DATA["Q28"]+DATA["Q32"])/8,1)
            
        DATA["affiliative"].replace(affiliative,inplace=True)
        DATA["selfenhancing"].replace(selfenhancing,inplace=True)
        DATA["agressive"].replace(aggressive,inplace=True)
        DATA["selfdefeating"].replace(selfdefeating,inplace=True)
    
 

    #Secondary dataframe of relevant columns for study
    DATA_S = DATA[["affiliative","selfenhancing",
                           "agressive","selfdefeating","age","gender"]]
    
    #Labels and Encoding
    if 1: 
        
        #Age bin column
        age = np.array(DATA_S["age"])
        rounded_ages = 10*np.floor(age/10)
        age_bins = []
        age_labels = ["10-20","20-30","30-40","40-70"]
        for i in range(len(rounded_ages)):
            rounded_age = rounded_ages[i]
            if rounded_age >=10 and rounded_age < 20:
                age_bins.append(age_labels[0])
            elif rounded_age >=20 and rounded_age < 30:
                age_bins.append(age_labels[1])
            elif rounded_age >=30 and rounded_age < 40:
                age_bins.append(age_labels[2])
            elif rounded_age >=40 and rounded_age < 50:
                age_bins.append(age_labels[3])
            elif rounded_age >=50 and rounded_age < 71:
                age_bins.append(age_labels[3])
            else: 
                continue
            
        gender_labels = np.where(gender==1,"male","female")
        age_bin_and_gender = list(zip(age_bins,gender_labels))
        labels = [tup[0]+","+tup[1] for tup in age_bin_and_gender]
        labels = np.array(labels).reshape(len(labels))
        
        #age_bin_and_gender = np.asarray(age_bin_and_gender)
        
        if 1: 
            if encoding == "one-hot":
                encoder = preprocessing.LabelBinarizer()
                encoded_targets = encoder.fit_transform(labels)
            else:
                encoder = preprocessing.LabelEncoder()
                encoded_targets = encoder.fit_transform(labels)
       
 
        #Done!
        
        
    # Set to 1 if need be
    #PCA plot
    if 0:
    
        fig,ax = plt.subplots()
        ax.set_xlabel("Number of PCA components",fontsize=20)
        ax.set_ylabel("Cumulative variance",fontsize=20)
        
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')  
        ax.grid()
        
        
        for n in range(1,5):
            
            pca = PCA(n_components=n)
            pca.fit(DATA_S[["affiliative","selfenhancing","agressive","selfdefeating"]])
            print(np.sum(pca.explained_variance_ratio_))
            ax.bar(n,np.sum(pca.explained_variance_ratio_),width=0.4)
        ax.set_xticks(ticks=np.arange(0,5),fontsize=15); ax.set_xticklabels(labels=ax.get_xticks(),fontsize=15)
        ax.set_yticks(ticks=np.arange(0,1.2,0.2),fontsize=15);ax.set_yticklabels(labels=ax.get_yticks(),fontsize=15)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        sys.exit()
     
    #Dataset info
    if 0:
        head = DATA_S.head()
        info = DATA_S.info()
        descr = DATA_S.describe()
    
    #Kernel density plots
    if 0:
        fig,ax = plt.subplots(figsize=(6,5))
        sns.kdeplot(x="affiliative",data=DATA_S,ax=ax,label="Affiliative")
        sns.kdeplot(x="selfenhancing",data=DATA_S,ax=ax,label="Self-Enhancing")
        sns.kdeplot(x="agressive",data=DATA_S,ax=ax,label="Aggressive")
        sns.kdeplot(x="selfdefeating",data=DATA_S,ax=ax,label="Self-Defeating")
        
        ax.legend()
        ax.set_xlabel("H-Type score")
        ax.set_ylabel("Kernel density estimate")
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')  
    
    if 0: #Gauging age distribution
        plot_dist(np.array(DATA_S["age"]))
    
    if 0: #Age bin distribution
        fig,ax = plt.subplots()
        ax.bar(np.arange(0,len(np.unique(labels))),np.unique(labels,return_counts=True)[1],color="r",alpha=0.8)
        fig.suptitle("Age bin distribution",fontsize=20,fontweight="bold")
        mod = [0]+list(np.unique(labels))
        ax.set_xticklabels(labels=mod,fontsize=15,rotation=315)
        ax.set_yticklabels(labels=ax.get_yticks(),fontsize=15)
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.set_xlabel("Classez",fontsize=20)
        ax.set_ylabel("Class Frequency",fontsize=20)
        
        
    #Normalisation
    if normalisation == True:  
        for col in DATA_S.columns[:4]:
            scaler = preprocessing.MinMaxScaler()
            column = np.expand_dims(DATA_S[col],axis=1)
            scaler.fit(column)
            column = scaler.transform(column)
            DATA_S[col]=column

    features = DATA_S.iloc[:,0:4]
    age = np.array(DATA_S.iloc[:,4])
    
    #PCA
    if PCA_features==True: 
        PCA = sklearn.decomposition.PCA(n_components=2)
        dimreduc_features = PCA.fit_transform(features)

    if polynomial_features == True: #Poly features
    
        
        poly_features = preprocessing.PolynomialFeatures(degree=2)
        features = poly_features.fit_transform(features)

    if Class_balancing==True: #Class Balancing
    
        rus = RandomUnderSampler(sampling_strategy="all")
        rebalanced_features, rebalanced_targets = rus.fit_resample(features,encoded_targets)
        X_train,X_test,Y_train,Y_test = train_test_split(rebalanced_features,rebalanced_targets,test_size=0.2)
        
    
    if 1: #Train test split
        
        X_train,X_test,Y_train,Y_test = train_test_split(features,encoded_targets,test_size=0.2)
        
    
    
    if 0: #Class frequency plot
        fig,ax = plt.subplots()
        ax.bar(np.arange(0,len(np.unique(encoded_targets))),np.unique(encoded_targets,return_counts=True)[1],color="r",alpha=0.8)
        fig.suptitle("Age bin distribution",fontsize=20,fontweight="bold")
        mod = [0]+list(np.unique(labels))
        ax.set_xticklabels(labels=mod,fontsize=15,rotation=315)
        ax.set_yticklabels(labels=[],fontsize=15)
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.set_xlabel("Classes",fontsize=20)
        ax.set_ylabel("Class Frequency",fontsize=20)

    

    if train_weight_calculation==True:
        

        classes = encoder.inverse_transform(Y_train)
        #Setting sample weights = accuracy*class_weight
        train_inds = X_train.index #Indices of training data
        response_accuracy = DATA.accuracy[train_inds]
        response_accuracy = np.array(response_accuracy)/100
        classes = encoder.inverse_transform(Y_train)
        targets,counts = np.unique(classes,return_counts=True)
        class_weights = 1/counts
        cws = np.zeros(len(Y_train))
        for clss in targets: 
            i = list(targets).index(clss)
            inds = classes==clss
            cws[inds]=class_weights[i]
    
        train_sample_weights = np.multiply(response_accuracy,cws)
        #train_sample_weights = cws
        #To check if weighting works:
        weighted_age = np.multiply(DATA_S.age[train_inds],cws)

    
#%%
#Initial model training 

#Model setup
if 1: 

 
        
    hyper_params = {}
    hyper_params["Ridge"] = {"alphas":np.logspace(-5,3,9),"cv":5}
    
    model_setup = {}
    
    
    #model_setup["Ridge"] = linear_model.RidgeClassifierCV(alphas = hyper_params["Ridge"]["alphas"],cv = hyper_params["Ridge"]["cv"])
    model_setup["LogReg"] = linear_model.LogisticRegressionCV(multi_class="ovr",max_iter=500) #Lots of choice with hyperparameters, may be interesting for hyp opt
    model_setup["SVC_rbf"] = svm.SVC(kernel = "rbf") #Apparently faster than SVR for med-sized datasets, low dimension datasets
    model_setup["SVC_linear"] = svm.SVC(kernel = "linear")
    model_setup["Gaussian Bayes"] = naive_bayes.GaussianNB() 
    model_setup["Decision Tree"] = tree.DecisionTreeClassifier()
    model_setup["Bagging"] = ensemble.BaggingClassifier(base_estimator=linear_model.RidgeClassifierCV(),max_samples=0.5,max_features=0.5)
    model_setup["AdaBoost"] = ensemble.AdaBoostClassifier()
    model_setup["Random Forest"] = ensemble.RandomForestClassifier(n_estimators=100)
    model_setup["GradBoost"] = ensemble.GradientBoostingClassifier()
    
    
    

    from yellowbrick.model_selection import learning_curve
    
    model_results = {}
    
    for model_name in model_setup.keys():
        print("Training {}\n".format(model_name))
        
        model = model_setup[model_name]
        learning_curve(model,X_train,Y_train,scoring="accuracy")
        
        model = model_setup[model_name]
        model.fit(X_train,Y_train,sample_weight=train_sample_weights)
        
        Y_pred = np.round(model.predict(X_test)).astype("int")
        
        accuracy, f1, balanced_acccuracy = classification_evaluation(Y_test, Y_pred)
        
        model_results[model_name] = (accuracy,f1)
        
    print("Training complete")

    
#%%
#Taking through top performing models for hyperparameter optimisation using BayesSearch

#Ridge, SVR, Adaboost setup
if 1: 
    
    
    hyper_params_opt = {}
    hyper_params_opt["GradBoost"] = {
                                    "n_estimators":list(np.arange(30,100,10)),
                                    "loss":["deviance"],
                                    "learning_rate":np.logspace(-5,4,8),
                                    "criterion": []
                                    
                                    }
    
    
    hyper_params_opt["Random Forest"] = {"criterion":["gini","entropy"],
                                   "n_estimators":list(np.arange(30,100,10)),
                                   "max_depth":np.arange(1,4)
                                }
    
    hyper_params_opt["Boosting Decision Tree"] = {
                                    "n_estimators":list(np.arange(30,100,10)),
                                    "learning_rate":np.logspace(-5,2,8)
                                    
                                    }
                                    

    chosen_models = [("GradBoost", ensemble.GradientBoostingClassifier()),
                     ("Random Forest",ensemble.RandomForestClassifier()),
                     ("Boosting Decision Tree",ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier())),
                     ]


#Hyperparameter optimisation  
if 1:     
    
        hyper_params_opt_results = {}
        
        for model in chosen_models[1:3]:
            
            #if model[0]!="AdaBoost":
            #continue
            
            print("Optimising hyperparameters for {} model\n\n".format(model[0]))
            print(hyper_params_opt[model[0]])
            hyper_params_opt_results[model[0]] = hyperparameter_optimisation(model[1], hyper_params_opt[model[0]], 
                                                                                      X_train,Y_train,search_type="Bayes",sample_weights=train_sample_weights)
        

        print("Model {} optimised \n\n".format(model[0]))
        
#%%


#Functions for model evaluation and comparison
#Learning curves for three models (analyse convergence, overfitting and underfitting
    

    
#Initialised optimised models
if 1: 
    model_test = []
    model_test.append(linear_model.RidgeCV(alphas=np.logspace(-4,3,8)))
    model_test.append(svm.SVC(kernel="poly",degree=3,
                        C=10))
    model_test.append(ensemble.AdaBoostRegressor(base_estimator=linear_model.ElasticNetCV(alphas=[1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]),
                                  loss="linear",
                                  n_estimators = 100))


if 1: #Model visualisation interface

    from yellowbrick.model_selection import LearningCurve
    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.classifier import ROCAUC
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.classifier import ClassPredictionError
    from yellowbrick.classifier import ConfusionMatrix
    from yellowbrick.model_selection import FeatureImportances
    from yellowbrick.style import set_palette
    from yellowbrick.style.palettes import PALETTES, SEQUENCES
    
    custom_palette = PALETTES["yellowbrick"]+PALETTES["flatui"][0:2]
    PALETTES["custom"] = custom_palette
    
    set_palette("custom",n_colors=8)
    
    def plot_editor(ax,legend=False,xlim=None,ylim=None,xlabel=None,ylabel=None,xticklabels=None,yticklabels=None):

        
        if legend==True:
            ax.legend(fontsize=15)
            
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        
        if xticklabels is not None:
            ax.set_xticklabels(labels=xticklabels,fontsize=14,rotation=315)
        if yticklabels is not None:
            ax.set_yticklabels(labels=yticklabels, fontsize=14)
        if xlim is not None:
            ax.set_xlim(xlim[0],xlim[1])
        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])
          
        if xlabel is not None:
            ax.set_xlabel(xlabel,fontsize=15)
        if ylabel is not None:
            ax.set_ylabel(ylabel,fontsize=15)
            
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        
    figure, axs = plt.subplots(2,2,sharex=True);
    axs = axs.reshape(axs.size,1)
    
    models = [ensemble.RandomForestClassifier(criterion="entropy",max_depth=3,n_estimators=100),
              ensemble.AdaBoostClassifier(n_estimators=100,learning_rate=1e-5),
              ensemble.GradientBoostingClassifier(n_estimators=100,learning_rate=1e-5),
              ensemble.AdaBoostClassifier(n_estimators=250)]
    final_model_names = ["Random Forest","AdaBoost n_est=100","Gradient Boost", "AdaBoost n_est=250"]
    
    plots = ["Training Curve"]
    cv = 20
    scoring = "f1_weighted"
    
    i=0
    for model in models:
        
        
        print(i)
        
        if "Training Curve" in plots:
            viz = LearningCurve(model,cv=cv,scoring=scoring,ax=axs[i][0],sample_weight=train_sample_weights)
            viz.fit(X_train,Y_train)
            
        if "Classification Report" in plots:
            viz= ClassificationReport(model,classes=np.unique(labels),support=True,ax=axs[i][0])
            viz.fit(X_train,Y_train)
            
            viz.score(X_test,Y_test)
            
        if "ROC-AUC" in plots:
            viz = ROCAUC(model,classes=np.unique(labels),ax=axs[i][0])
            viz.fit(X_train,Y_train)
            viz.score(X_test,Y_test)
            
        if "Precision Recall" in plots:
            viz = PrecisionRecallCurve(model,per_class=True,ax=axs[i][0],sample_weight=train_sample_weights)
            viz.fit(X_train,Y_train)
            viz.score(X_test,Y_test)
            
        if "Class Prediction Error" in plots:
            viz = ClassPredictionError(model, classes=np.unique(labels),ax=axs[i][0])
            viz.fit(X_train,Y_train,sample_weight=train_sample_weights)
            viz.score(X_test,Y_test)
            print(Y_test.shape)

            
        if "Confusion Matrix" in plots:
            viz = ConfusionMatrix(model,classes=np.unique(labels))
            viz.fit(X_train,Y_train)
            viz.score(X_test,Y_test)
            plot_editor(viz.ax)
            viz.show()
            
        if "Feature Importance" in plots:
            viz = FeatureImportances(model,labels=["affiliative","selfenhancing",
                                   "agressive","selfdefeating"],relative=False)
            viz.show()
            viz.fit(X_train,Y_train)
            
        #Axes editing  
        ax = axs[i][0]
        #if i != 3: 
        #ax.get_legend().set_visible(False)
        
        if i==0:
            plot_editor(ax,legend=False, 
                        xticklabels=np.unique(labels),
                        yticklabels=ax.get_yticks(),
                        ylabel="Weighted F1",
                        ylim=(0,1))
            
        if i==1:
            plot_editor(ax,legend=False, 
                        xticklabels=np.unique(labels),
                        yticklabels=ax.get_yticks(),
                        xlabel="Weighted F1",
                        ylim=(0,1))
            
        if i==2:

            plot_editor(ax,legend=False, 
                        xticklabels=ax.get_xticks(),
                        yticklabels=ax.get_yticks(),
                        ylabel="Weighted F1",
                        xlabel="Training Samples",
                        ylim=(0,1))
            
        if i==3:
            plot_editor(ax,legend=False, 
                        xticklabels=ax.get_xticks(),
                        yticklabels=ax.get_yticks(),
                        xlabel="Training Samples",
                        ylim=(0,1))
        
        ax.text(0.4,0.8,final_model_names[i],fontsize=15,transform=ax.transAxes)
        i+=1
        

    
    
    
    figure.suptitle("Training-Blue || Validation-Green",fontweight="bold",fontsize=20)
    figure.tight_layout()
        
        
    

            






















