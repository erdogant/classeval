import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
import classeval as cle


def test_summary():
    X, y = cle.load_example('breast')
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prediction
    model = clf.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # CHECK summary two class
    results = cle.eval(y_true, y_proba[:,1], pos_label='malignant')
    assert np.all(results['confmat']['confmat']==[[69,2],[3,40]])
    assert results['f1'].astype(str)[0:4]=='0.94'
    assert results['auc'].astype(str)[0:4]=='0.99'
    assert results['kappa'].astype(str)[0:4]=='0.90'
    assert results['MCC'].astype(str)[0:5]=='0.906'
    assert results['average_precision'].astype(str)[0:4]=='0.99'
    assert results['CAP'].astype(str)=='43'

    # CHECK using bool as input
    results = cle.eval(y_true=='malignant', y_proba[:,1])
    assert np.all(results['confmat']['confmat']==[[69,2],[3,40]])
    assert results['f1'].astype(str)[0:4]=='0.94'
    assert results['auc'].astype(str)[0:4]=='0.99'
    assert results['kappa'].astype(str)[0:4]=='0.90'
    assert results['MCC'].astype(str)[0:5]=='0.906'
    assert results['average_precision'].astype(str)[0:4]=='0.99'
    assert results['CAP'].astype(str)=='43'

    # CHECK dict output
    assert np.all(np.isin([*results.keys()], ['class_names', 'pos_label', 'neg_label', 'y_true', 'y_pred', 'y_proba', 'auc', 'f1', 'kappa', 'report', 'thresholds', 'fpr', 'tpr', 'average_precision', 'precision', 'recall', 'MCC', 'CAP', 'TPFP', 'confmat', 'threshold']))

    # CHECK plots
    ax = cle.plot(results)

    # TEST 2: Check model output is unchanged
    X, y = cle.load_example('iris')
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=1)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_score = model.decision_function(X_test)

    # CHECK confmat
    out = cle.confmatrix.eval(y_true, y_pred, normalize=True)
    assert np.all(out['confmat'].astype(str)==[['1.0', '0.0', '0.0'], ['0.0', '0.9230769230769231', '0.07692307692307693'], ['0.0', '0.0', '1.0']])
    out = cle.confmatrix.eval(y_true, y_pred, normalize=False)
    assert np.all(out['confmat']==[[11, 0, 0], [0, 12, 1], [0, 0, 6]])

    results = cle.eval(y_true, y_proba, y_score, y_pred)

    # CHECK output
    assert np.all(np.isin([*results.keys()], ['y_true', 'y_pred', 'y_proba', 'threshold', 'class_names', 'ROCAUC', 'stackbar', 'confmat']))
    # assert results['ROCAUC']['auc'].astype(str)[0:4]=='0.98'

    # CHECK plot
    ax = cle.plot(results)
