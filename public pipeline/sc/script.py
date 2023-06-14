import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def filter_data(df):
    data = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return data.drop(columns_to_drop, axis=1)


def remove(df):
    data = df.copy()
    q25 = data['year'].quantile(0.25)
    q75 = data['year'].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    data.loc[data['year'] < boundaries[0], 'year'] = round(boundaries[0])
    data.loc[data['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return data


def short_model(data):
    if not pd.isna(data):
        return data.lower().split(' ')[0]
    else:
        return data


def create_short_model(df):
    data = df.copy()
    data['short_model'] = data['model'].apply(short_model)
    return data


def create_age_category(df):
    data = df.copy()
    data.loc[:, 'age_category'] = data['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return data


def main():
    print('Loan Prediction Pipeline')

    df = pd.read_csv('data/PATH_YOUR_DATA.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=['object'])

    preprocessor_data = Pipeline(steps=[
        ('filter_data', FunctionTransformer(filter_data)),
        ('remove', FunctionTransformer(remove)),
        ('create_short_model', FunctionTransformer(create_short_model)),
        ('create_age_category', FunctionTransformer(create_age_category)),
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor_data', preprocessor_data),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'loan_pipe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
