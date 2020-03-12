# %%
import classeval as clf
print(clf.__version__)


# %% Import example dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()


# %%
X, y = clf.load_example('breast')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% Two-class

# Prediction
model = gb.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

results = clf.summary(y_test, y_proba[:,1])
print(results['report'])

out = clf.confmatrix.eval(y_test, y_pred, normalize=True)
clf.confmatrix.plot(out)
out = clf.confmatrix.eval(y_test, y_pred, normalize=False)
clf.confmatrix.plot(out)

# CAP
results_CAP = clf.CAP(y_test, y_proba[:,1])
# MCC
results_MCC = clf.MCC(y_test, y_proba[:,1])
# MCC
results_proba = clf.proba_curve(y_test, y_proba[:,1])


# %% Multi-class
X, y = clf.load_example('iris')

y=y.astype(str)
y[y=='0']='iris'
y[y=='1']='bla'
y[y=='2']='tulip'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = gb.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

out = clf.confmatrix.eval(y_test, y_pred, normalize=True)
clf.confmatrix.plot(out)
out = clf.confmatrix.eval(y_test, y_pred, normalize=False)
clf.confmatrix.plot(out)

print(results['report'])
