import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()


def test_summary():
    X, y = classeval.load_example('breast')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prediction
    model = gb.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # summary two class
    results = classeval.summary(y_test, y_proba[:,1], showfig=False)
    assert np.all(results['confmat']['confmat']==[[40,3],[2,69]])
    assert results['f1'].astype(str)[0:4]=='0.96'
    assert results['auc'].astype(str)[0:4]=='0.99'
    assert results['kappa'].astype(str)[0:4]=='0.90'
    assert results['MCC'].astype(str)[0:5]=='0.906'
    assert results['average_precision'].astype(str)[0:5]=='0.996'
    assert results['CAP'].astype(str)=='71'


    # TEST 2: Check model output is unchanged
    X, y = classeval.load_example('iris')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = gb.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    out = classeval.confmatrix.eval(y_test, y_pred, normalize=True)
    assert np.all(out['confmat'].astype(str)==[['1.0', '0.0', '0.0'], ['0.0', '0.9230769230769231', '0.07692307692307693'], ['0.0', '0.0', '1.0']])
    out = classeval.confmatrix.eval(y_test, y_pred, normalize=False)
    assert np.all(out['confmat']==[[11, 0, 0], [0, 12, 1], [0, 0, 6]])
