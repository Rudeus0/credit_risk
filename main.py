from src.train import (data_load, features, transform_scaler, credit_model)
from src.evaluate import credit_evaluate


if __name__ == "__main__":
    
    credit = data_load()
    
    credit_dum = features(credit)
    
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = transform_scaler(credit_dum)
    
    model_lgr, model_rfc, model_xgb = credit_model(X_train_scaled, y_train)
    
    credit_evaluate(X_test_scaled, y_test, model_lgr, model_rfc, model_xgb)
    
    
    