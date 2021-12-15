import pandas as pd, numpy as np
import time
from sklearn import  metrics




def main():
    X_test = pd.read('../../data/interim/test_set.csv')
    X_test_scaled = pd.read('../../data/interim/test_set_scaled.csv')
    
    results_file = open( '../../reports/results','w')

    for p in []:
        with open(f"../../data/interim/{p}", 'rb') as f:
            clf = pickle.load(f)
        t0 = time.time()
        predicted = clf.predict(X_test)
        print("\nModel: "+p, file=results_file)
        print(time.time()-t0, file=results_file)
        print( metrics.classification_report(y_test, predicted), file=results_file )

    close(results_file)
        

if __name__ == '__main__':
    main()