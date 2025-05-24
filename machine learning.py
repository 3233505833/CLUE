import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr, spearmanr

def preprocess_data(df):
    X = df
    X = X.astype(float)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y_classification = df['gold_label']
    X_train, X_test, y_train_class, y_test_class, indices_train, indices_test = train_test_split(
        X, y_classification, df.index, test_size=0.5, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train_class, y_test_class, indices_test

def find_best_params(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    def pearson_scorer(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]

    pearson_scorer = make_scorer(pearson_scorer)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=pearson_scorer,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    return grid_search.best_estimator_

def predict(model, X_test):
    predictions = model.predict(X_test)
    print(predictions)
    return predictions

# 主流程
if __name__ == "__main__":
    df = pd.read_csv('ml/data_for_machinelearning.csv')
    X_train_scaled, X_test_scaled, y_train_class, y_test_class, indices_test = preprocess_data(df)
    best_model = find_best_params(X_train_scaled, y_train_class)
    classification_predictions = predict(best_model, X_test_scaled)
