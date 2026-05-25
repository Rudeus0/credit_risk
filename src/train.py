import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_load()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit

def features(credit: pd.DataFrame)-> pd.DataFrame:
    credit['class'] = credit['class'].map({"good": 0, "bad": 1})
    categorical_cols = credit.select_dtypes(include=["object", "str"]).columns.to_list()
    
    if 'class' in categorical_cols:
        categorical_cols.remove('class')
        
    credit_dum = pd.get_dummies(credit, columns=categorical_cols, drop_first=True)
    return credit_dum

def transform_scaler(credit_dum: pd.DataFrame)-> tuple:
    X = credit_dum.drop(['class'], axis=1)
    y = credit_dum['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
     

    
    