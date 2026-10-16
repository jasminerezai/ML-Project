import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    X = df.drop(columns=['Species']).values
    y_raw = df['Species'].values
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, encoder