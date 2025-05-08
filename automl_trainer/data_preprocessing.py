import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return preprocessor, X, y
