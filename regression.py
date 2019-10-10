import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
bos = pd.read_csv('datasets/data/housing_boston.csv', header=None, delimiter=r"\s+", names=column_names)
random_state = 12


def model_selection_linear():
    x, y = feature_selection()
    print(x.head())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    pipelines = []
    pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
    pipelines.append(('Ridge', Pipeline([('Scaler', StandardScaler()), ('Ridge', Ridge())])))
    pipelines.append(('LASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
    pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
    pipelines.append(('SVR', Pipeline([('Scaler', StandardScaler()), ('SVR', SVR())])))
    results = []
    names = []
    for name, model in pipelines:
        kfold = KFold(n_splits=10, random_state=random_state)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def model_selection_ensemble():
    ensembles = []
    x, y = feature_selection()
    print(x.head())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    ensembles.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
    ensembles.append(('AB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
    ensembles.append(('GBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
    ensembles.append(('RF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
    ensembles.append(('ET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))
    results = []
    names = []
    for name, model in ensembles:
        kfold = KFold(n_splits=10, random_state=random_state)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def feature_selection():
    X = bos.drop(['CRIM', 'ZN', 'INDUS', 'NOX', 'AGE', 'DIS', 'RAD', 'MEDV'], axis=1)
    Y = bos["MEDV"]
    return X, Y


def plot_corr():
    correlations = bos.corr()
    sns.heatmap(correlations, square=True, cmap="YlGnBu")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()


def plot_linear_reg():
    x, y = feature_selection()
    print(x.head())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)
    lm = Ridge()
    model = lm.fit(x_train, y_train)
    pred_y = lm.predict(x_test)
    pd.DataFrame({"Actual": y_test, "Predict": pred_y}).head()
    plt.scatter(y_test, pred_y)
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()
    mse = mean_squared_error(y_test, pred_y)
    print(mse)


if __name__ == '__main__':
    plot_corr()
    plot_linear_reg()
    model_selection_linear()
    model_selection_ensemble()
