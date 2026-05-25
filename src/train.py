import pandas as pd 

def data_load()-> pd.DataFrame:
    credit = pd.read_csv("../data/credit_customers.csv")
    return credit

