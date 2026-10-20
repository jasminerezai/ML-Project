import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    # Load CSV into a Pandas DataFrame
    df = pd.read_csv(file_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Separate features (X) from class labels (y)
    X = df.drop(columns=['Species']).values
    y_raw = df['Species'].values

    # Convert text species names into integers (0, 1, 2)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21, stratify=y
    )

    # Apply Z-score standardization: (x - mean) / std
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, encoder