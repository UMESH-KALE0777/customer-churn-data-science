from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    
