import pandas as pd, numpy as np
import os
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import preprocessing, tree
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from pathlib import Path


def decision_tree(X_train, y_train, to_save_dir):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    with open(os.path.join( to_save_dir,'dec_tree.pkl'), 'wb') as f:
        pickle.dump(clf, f)

def random_forest(X_train, y_train, to_save_dir):
    clf_rf = RandomForestClassifier(n_estimators=25, max_depth=20, random_state=0, warm_start=True, min_samples_leaf=2, 
                                    max_features='sqrt', min_samples_split=10)
    clf_rf.fit(X_train, y_train)
    with open(os.path.join( to_save_dir,'random_forest.pkl'), 'wb') as f:
        pickle.dump(clf_rf, f)

def svm(X_train, y_train, to_save_dir):
    clf_svm = make_pipeline(preprocessing.StandardScaler(), SVC(gamma=0.01,degree=3, kernel='rbf'))
    clf_svm.fit(X_train, y_train)
    with open(os.path.join( to_save_dir,'svm.pkl'), 'wb') as f:
        pickle.dump(clf_svm, f)

def gradient_boost(X_train, y_train, to_save_dir):
    clf_gb = GradientBoostingClassifier(n_estimators=25, learning_rate=1.0,
                                        max_depth=20, random_state=0)
    clf_gb.fit(X_train, y_train)
    with open(os.path.join( to_save_dir,'gradient_boost.pkl'), 'wb') as f:
        pickle.dump(clf_gb, f)

def neural_net_mlp(X_train_scaled, y_train, to_save_dir):

    clf_mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(14, 10 ), random_state=1, max_iter=300, warm_start=True)
    clf_mlp.fit(X_train_scaled, y_train) 
    with open(os.path.join( to_save_dir, 'mlp.pkl'), 'wb') as f:
        pickle.dump(clf_mlp, f)

def ensemble(X_train_scaled, y_train, to_save_dir):
    clf1 = RandomForestClassifier(n_estimators=25, max_depth=20, random_state=0, warm_start=True, min_samples_leaf=2, 
                                    max_features='sqrt', min_samples_split=10)
    clf2 = GradientBoostingClassifier(n_estimators=25, learning_rate=1.0,max_depth=20, random_state=0)
    clf3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(14 , 10), random_state=1, max_iter=300, warm_start=True)    
    
    #voting classifier
    ecl_voting = VotingClassifier(estimators=[('rf', clf1), ('grb', clf2), ('mlp', clf3)], weights=[1,1,1], voting='hard')
    ecl_voting.fit(X_train_scaled, y_train)
    with open(os.path.join( to_save_dir,'ensemble_voting.pkl'), 'wb') as f:
        pickle.dump(ecl_voting, f)

    #stacking classifier
    ecl_stacking = StackingClassifier(estimators=[('rf', clf1), ('grb', clf2), ('mlp', clf3)])
    ecl_stacking.fit(X_train_scaled, y_train)
    with open( os.path.join( to_save_dir,'ensemble_stacking.pkl'), 'wb') as f:
        pickle.dump(ecl_stacking, f)


def resample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy=0.5)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    undersample = RandomUnderSampler(sampling_strategy=0.6)
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    return X_train, y_train

def main():
    file_dir = os.path.dirname(__file__) 
    logger = logging.getLogger(__name__)


    for i in ['0','30','60','90','120']: # iterate through different doc vec feature sizes
        logger.info(i)
        models_dir = os.path.join( file_dir, '..','..','models', i)
        #create directories if they don't exist
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(file_dir,'..','..','data','interim',i,)).mkdir(parents=True, exist_ok=True)


        df_combined = pd.read_csv( os.path.join(file_dir,'..','..','data', 'processed',f'features_with_doc_vector_{i}.csv'), index_col=0)

        X_train, X_test, y_train, y_test = train_test_split(df_combined.drop('anomalous',1), df_combined['anomalous'], 
                                                        test_size=0.3, random_state=42, shuffle = True)
        

        #create scaled data, to be used for some models
        scaler = preprocessing.StandardScaler() 
        scaler.fit(X_train)  
        X_train_scaled = scaler.transform(X_train) 
        X_test_scaled = scaler.transform(X_test)

        #save scaled test sets, to be used later for testing
        X_test.to_csv(os.path.join(file_dir,'..','..','data','interim',i,'test_set.csv'))
        pd.DataFrame(X_test_scaled).to_csv(os.path.join(file_dir,'..','..','data','interim',i,'test_set_scaled.csv'))

        #create and savemodels
        decision_tree(X_train, y_train, models_dir)
        random_forest(X_train, y_train, models_dir)
        svm(X_train, y_train, models_dir)
        gradient_boost(X_train, y_train, models_dir)
        neural_net_mlp(X_train_scaled, y_train, models_dir)
        #ensemble(X_train_scaled, y_train, models_dir)

    y_test.to_csv( os.path.join(file_dir,'..','..','data','processed','y_test.csv') )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()