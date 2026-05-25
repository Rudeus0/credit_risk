import pandas as pd 

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