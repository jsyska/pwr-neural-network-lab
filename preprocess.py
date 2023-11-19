from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pandas.core.series import Series


WEEKDAYS_MAP = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS_MAP = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4,
    "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9,
    "nov": 10, "dec": 11,
}


def preprocess_heart_disease():
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv('data/processed/X_train_heart_disease.csv', index=False)
    y_train.to_csv('data/processed/y_train_heart_disease.csv', index=False)
    X_test.to_csv('data/processed/X_test_heart_disease.csv', index=False)
    y_test.to_csv('data/processed/y_test_heart_disease.csv', index=False)


def preprocess_heart_disease_from_csv():
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    df = pd.read_csv('data/raw/heart_disease/processed.cleveland.data', names=column_names, na_values='?')

    df.fillna(df.mean(), inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv('data/processed/X_train_heart_disease.csv', index=False)
    y_train.to_csv('data/processed/y_train_heart_disease.csv', index=False)
    X_test.to_csv('data/processed/X_test_heart_disease.csv', index=False)
    y_test.to_csv('data/processed/y_test_heart_disease.csv', index=False)


def preprocess_affnist():
    pass


def preprocess_age_prediction():
    pass


def _process_forest_element(val: str | float) -> str | float:
    if not isinstance(val, str):
        return val

    if val in WEEKDAYS_MAP:
        return WEEKDAYS_MAP[val]

    if val in MONTHS_MAP:
        return MONTHS_MAP[val]

    raise Exception("Value not matched")


def _process_forest_row(series: Series) -> Series:
    return series.map(_process_forest_element)


def preprocess_forest_fires() -> None:
    area_header = "area"
    df = pd.read_csv('data/raw/forest_fires/forestfires.csv')
    df = df.apply(_process_forest_row)
    df = df.astype(float)

    # Normalize the outputs
    y = df[[area_header]] / df[[area_header]].max()
    X = df.drop(area_header, axis=1)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save to CSV
    X_train.to_csv('data/processed/X_train_forest_fires.csv', index=False)
    y_train.to_csv('data/processed/y_train_forest_fires.csv', index=False)
    X_test.to_csv('data/processed/X_test_forest_fires.csv', index=False)
    y_test.to_csv('data/processed/y_test_forest_fires.csv', index=False)

