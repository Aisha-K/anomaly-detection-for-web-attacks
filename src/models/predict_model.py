import pandas as pd, numpy as np
import time
from sklearn import  metrics
import pickle


import os

# predicts results for x_test, compares against y_test, and prints to file results_file
def model_predict(model_dir, model_name , results_file, X_test, y_test):
    with open( os.path.join(model_dir, model_name ) , 'rb') as f:
        clf = pickle.load(f)
    t0 = time.time()
    predicted = clf.predict(X_test)
    print("\nModel: "+model_name, file=results_file)
    print(time.time()-t0, file=results_file)
    print( metrics.classification_report(y_test, predicted), file=results_file )    

def main():
    file_dir = os.path.dirname(__file__) 
    y_test = pd.read_csv( os.path.join(file_dir,'..','..','data','processed','y_test.csv') , index_col=0 )

    for m_dir in ['0', '30','60','90','120']:
        models_dir = os.path.join( file_dir, '..','..','models', m_dir)

        #read in test data
        X_test = pd.read_csv(os.path.join(file_dir,'..','..','data','interim',m_dir,'test_set.csv'), index_col=0)
        X_test_scaled = pd.read_csv(os.path.join(file_dir,'..','..','data','interim',m_dir,'test_set_scaled.csv'), index_col =0)
        
        results_file = open( os.path.join( file_dir, '..','..','reports',f'results_{m_dir}.txt'),'w')

        #for each model that uses non scaled data
        for p in ['dec_tree.pkl', 'random_forest.pkl', 'svm.pkl', 'gradient_boost.pkl']:
            model_predict(models_dir,p, results_file, X_test, y_test)

        #for each model that uses scaled data
        for p in ['mlp.pkl']: #, 'ensemble_voting.pkl', 'ensemble_stacking.pkl']:
        #for p in ['mlp.pkl', 'ensemble_voting.pkl', 'ensemble_stacking.pkl']:
            model_predict(models_dir,p, results_file, X_test_scaled, y_test)

        results_file.close()
        

if __name__ == '__main__':
    main()