import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Handling missing values
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Convert categorical variables to numeric using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    return data

def explore_data(data):
    # Handle missing values
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Convert non-numeric columns to numeric using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Display head of the dataset
    print("Head of the dataset:")
    print(data.head())

    # Display info of the dataset
    print("\nInfo of the dataset:")
    print(data.info())

    # Identify non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()

    # EDA - Exploratory Data Analysis
    # Heatmap
    numeric_data = data.drop(columns=non_numeric_cols, errors='ignore')  # Exclude non-numeric columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    # Distribution of Numerical Variables
    data.hist(figsize=(15, 10), bins=20)
    plt.suptitle('Distribution of Numerical Variables', x=0.5, y=1.02, fontsize=16)
    plt.show()

    # Boxplot of Numerical Variables
    data.plot(kind='box', figsize=(15, 10), vert=False)
    plt.title('Boxplot of Numerical Variables')
    plt.show()

    # Identify Outliers
    outliers = []
    for column in numeric_data.columns:
        Q1 = numeric_data[column].quantile(0.25)
        Q3 = numeric_data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_indices = numeric_data[(numeric_data[column] < Q1 - 1.5 * IQR) | (numeric_data[column] > Q3 + 1.5 * IQR)].index
        outliers.extend(outlier_indices)

    outliers = list(set(outliers))
    print("Number of outliers:", len(outliers))

    # Remove outliers
    data_cleaned = data.drop(outliers, axis=0).reset_index(drop=True)
    print("Shape of data after removing outliers:", data_cleaned.shape)

    # Pairplot (after removing outliers)
    sns.pairplot(data_cleaned.sample(500), hue='income_>50K', diag_kind='kde', palette='husl', vars=['age', 'education.num', 'hours.per.week'])
    plt.suptitle('Pairplot of Selected Features (After Removing Outliers)', y=1.02)
    plt.show()

    return data_cleaned

def preprocess_and_explore(file_path):
    # Load data
    data = load_data(file_path)

    # Explore data
    data_cleaned = explore_data(data)

    # Preprocess data
    data_cleaned = preprocess_data(data_cleaned)

    # Split the data into features and target
    X_train, X_test, y_train, y_test = preprocess_data(data_cleaned)

    # Train models
    logistic_pipeline, knn_pipeline, dt_pipeline = train_model(X_train, y_train)

    # Evaluate models
    evaluate_model(logistic_pipeline, X_test, y_test)
    evaluate_model(knn_pipeline, X_test, y_test)
    evaluate_model(dt_pipeline, X_test, y_test)

def preprocess_data(data):
    # Split the data into features and target
    X = data.drop('income_>50K', axis=1)
    y = data['income_>50K']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Logistic Regression
    logistic_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    logistic_pipeline.fit(X_train, y_train)

    # K-Nearest Neighbors
    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
    knn_pipeline.fit(X_train, y_train)

    # Decision Tree
    dt_pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    dt_pipeline.fit(X_train, y_train)

    return logistic_pipeline, knn_pipeline, dt_pipeline

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    print(model.steps[-1][0] + " Model:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    preprocess_and_explore('adult.csv')
