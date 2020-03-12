# %%
import classeval
print(classeval.__version__)


# %% Import example dataset
X,y = classeval.load_example()
X['y'] = y


# %%
# !pip install df2onehot
from sklearn.model_selection import train_test_split
import df2onehot

out = df2onehot.df2onehot(X)['numeric']
out.dropna(inplace=True)
y=out['y'].astype(float).values
del out['y']
[X_train, X_test, y_train, y_true]=train_test_split(out, y, test_size=0.2)


# %% libraries
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()

# Prediction
model = gb.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

results = classeval.summary(y_true, y_proba[:,1])

out = classeval.confmatrix.eval(y_true, y_pred, normalize=True)
out = classeval.confmatrix.eval(y_true, y_pred, normalize=False)
classeval.confmatrix.plot(out)
results_CAP = classeval.CAP(y_true, y_proba[:,1])


# %%
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

X, y_true = datasets.load_iris(return_X_y=True)
model = RandomForestClassifier(random_state=1).fit(X, y_true)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

out = classeval.confmatrix.eval(y_true, y_pred, normalize=True)
out = classeval.confmatrix.eval(y_true, y_pred, normalize=False)
classeval.confmatrix.plot(out)

results = classeval.summary(y_true==0, y_proba[:,0])
print(results['report'])

