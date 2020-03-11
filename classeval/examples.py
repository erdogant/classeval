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
[X_train, X_test, y_train, y_test]=train_test_split(out, y, test_size=0.2)


# %% libraries
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
gb = GradientBoostingClassifier()

# Prediction
model = gb.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)

results = classeval.summary(y_test, y_proba[:,1])
results_CAP = classeval.CAP(y_test, y_proba[:,1])
