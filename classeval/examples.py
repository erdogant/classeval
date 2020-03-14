# %%
import classeval as clf
print(dir(clf))
print(clf.__version__)


# %% Import example dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()


# %% Two-class
X, y = clf.load_example('breast')
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2)

# Prediction
model = gb.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

# ROC evaluation
out_ROC = clf.ROC.eval(y_true, y_proba, pos_label='malignant')
ax = clf.ROC.plot(out_ROC, title='Breast dataset')
# Its also OK to set the y_true as bool.
out_ROC = clf.ROC.eval(y_true=='malignant', y_proba)
ax = clf.ROC.plot(out_ROC, title='Breast dataset')

# Confmatrix evaluation
out_CONFMAT = clf.confmatrix.eval(y_true, y_pred, normalize=True)
clf.confmatrix.plot(out_CONFMAT, fontsize=18)
out_CONFMAT = clf.confmatrix.eval(y_true, y_pred, normalize=False)
clf.confmatrix.plot(out_CONFMAT)

# Total evaluation
out = clf.eval(y_true, y_proba, pos_label='malignant')
out = clf.eval(y_true=='malignant', y_proba)
ax = clf.plot(out, figsize=(20,15), fontsize=14)

# Some results
print(out['report'])


# %% Multi-class
X,y = clf.load_example('iris')
X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.5)

model = gb.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_score = model.decision_function(X_test)

# ROC evaluation
out_ROC = clf.ROC.eval(y_true, y_proba, y_score)
ax = clf.ROC.plot(out_ROC, title='Iris dataset')

# Confmatrix evaluation
out_CONFMAT = clf.confmatrix.eval(y_true, y_pred, normalize=True)
ax = clf.confmatrix.plot(out_CONFMAT)
out_CONFMAT = clf.confmatrix.eval(y_true, y_pred, normalize=False)
ax = clf.confmatrix.plot(out_CONFMAT)

out = clf.eval(y_true, y_proba, y_score, y_pred)
ax = clf.plot(out)


# %% Fin