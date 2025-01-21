import mlflow
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
import xgboost as xgb
import pickle


def main():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    label_mapping = {
        " Fair / Windy ": 3,
        " Fair ": 1,
        " Light Rain / Windy ": 7,
        " Partly Cloudy ": 2,
        " Mostly Cloudy ": 2,
        " Cloudy ": 5,
        " Light Rain ": 6,
        " Mostly Cloudy / Windy ": 8,
        " Partly Cloudy / Windy ": 5,
        " Light Snow / Windy ": 4,
        " Cloudy / Windy ": 5,
        " Light Drizzle ": 5,
        " Rain ": 6,
        " Heavy Rain ": 9,
        " Fog ": 8,
        " Wintry Mix ": 4,
        " Light Freezing Rain ": 8,
        " Light Snow ": 3,
        " Wintry Mix / Windy ": 4,
        " Fog / Windy ": 8,
        " Light Drizzle / Windy ": 6,
        " Rain / Windy ": 7,
        " Drizzle and Fog ": 9,
        " Snow ": 3,
        " Heavy Rain / Windy ": 10,
    }

    with mlflow.start_run(log_system_metrics=True, run_name="WeatherPrediction") as run:
        weather_data = pd.read_csv(config["weather_data"])
        weather_data["SafetyLevel"] = weather_data[" Condition "].map(label_mapping) -1
        features = [
            "Temperature",
            "Humidity",
            "Wind Speed",
            "Pressure",
        ]
        X = weather_data[features]
        y = weather_data["SafetyLevel"]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = xgb.XGBClassifier(config["weather_xgb_params"])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(clf, "model")

        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        print(classification_report(y_test, y_pred))

        with open("model/weather_model.pkl", "wb") as f:
            pickle.dump(clf, f)

    with mlflow.start_run(run_name="ServicePrediction") as run:
        service_data = pd.read_csv(config["service_data"])

        le = LabelEncoder()

        service_data["Warranty Status"] = le.fit_transform(
            service_data["Warranty Status"]
        )
        with open("model/warranty_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        service_data["Company"] = le.fit_transform(service_data["Company"])
        
        with open("model/company_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        X = service_data[
           ["Days Since Servicing", "Warranty Status", "Days Since Purchase", "Company"]
        ]
        y = service_data["Status"]
        
        y = le.fit_transform(y)
        with open("model/status_encoder.pkl", "wb") as f:
            pickle.dump(le, f)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = xgb.XGBClassifier(config["service_xgb_params"])
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(clf, "model")

        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        print(classification_report(y_test, y_pred))

        with open("model/service_model.pkl", "wb") as f:
            pickle.dump(clf, f)
            
if __name__ == "__main__":
    main()
        
        
