from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def credit_evaluate(X_test_scaled,y_test, model_lgr, model_rfc, model_xgb):
    
    for name, model in [("Logisitic Regression", model_lgr),
                         ("Random Forset Classification", model_rfc),
                         ("XGBoostr", model_xgb)]:
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        print(f"\n \n{name}")
        print(f"Accuracy: {acc:.4f} | ROC AUC: {roc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))