import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_models(df):
    class_column = df.columns.str.strip()
    if 'class' not in class_column.tolist():
        return {"error": "Dataset must contain 'class' column"}
    
    df.columns = class_column
    df['class'].replace(['bankruptcy','non-bankruptcy'], [0,1], inplace=True)
    x = df.drop(columns=['class'])
    y = df['class']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    return results

if __name__ == "__main__":
    print("This script is intended to be run in an interactive environment.")
