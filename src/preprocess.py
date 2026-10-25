import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Complete Exploratory Data Analysis Checks
    print("DATASET SUMMARY")
    print(f"Shape: {df.shape}")
    print("\nData Types & Non-Null Counts:")
    print(df.info())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nDescriptive Statistics:")
    print(df.describe().T)
    
    # Cleaning Decisions
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        
    drop_cols = [c for c in ['Id', 'id', 'ID'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        
    target_col = 'species' if 'species' in df.columns else 'Species'
    
    # Encode Categorical Feature Columns
    # Convert 'soil_type' into one-hot binary columns
    if 'soil_type' in df.columns:
        df = pd.get_dummies(df, columns=['soil_type'], drop_first=True, dtype=float)
    
    X = df.drop(columns=[target_col]).values
    y_raw = df[target_col].values
    
    # Stratified Split BEFORE Fitting Transformers
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, random_state=21, stratify=y_raw
    )
    
    # Fit LabelEncoder strictly on training targets
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)
    
    # Fit StandardScaler strictly on training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, X_train_scaled, X_test_scaled, y_train, y_test, encoder