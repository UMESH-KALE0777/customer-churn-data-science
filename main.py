

from src.preprocessing import preprocess_data
from src.train_model import train_model
from src.evaluate import evaluate_model

def main():
    X_train, X_test, y_train, y_test = preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
