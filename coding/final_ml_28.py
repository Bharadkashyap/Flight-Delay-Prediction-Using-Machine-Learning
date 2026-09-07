import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Flight_Delay_Prediction_Dataset_Raw.csv")

print("Dataset loaded successfully")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

# Check unique values

for col in ["Airline", "Weather", "Aircraft_Type", "Holiday", "Delay_Status"]:
    print(col, ":", df[col].unique())

# Check duplicates and target variable

print("Duplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["Delay_Status"].value_counts())
# Check numerical data

df.describe()

# Remove duplicate rows

duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates)
print("Rows after removing duplicates:", df.shape[0])

# Impute missing values

num_imputer = SimpleImputer(strategy="median")
df[["Temperature_C", "Wind_Speed_kmph", "Distance_km"]] = num_imputer.fit_transform(
    df[["Temperature_C", "Wind_Speed_kmph", "Distance_km"]]
)

cat_imputer = SimpleImputer(strategy="most_frequent")
df[["Airline", "Weather", "Aircraft_Type"]] = cat_imputer.fit_transform(
    df[["Airline", "Weather", "Aircraft_Type"]]
)

print("Missing values imputed")

# Check missing values after imputation

print(df.isnull().sum())

# Remove outliers

q1 = df["Wind_Speed_kmph"].quantile(0.25)
q3 = df["Wind_Speed_kmph"].quantile(0.75)
iqr = q3 - q1

df = df[
    (df["Wind_Speed_kmph"] >= q1 - 1.5 * iqr) &
    (df["Wind_Speed_kmph"] <= q3 + 1.5 * iqr)
].copy()

print("Outliers removed")
print("Rows after outlier removal:", df.shape[0])

# Encode categorical columns

le = LabelEncoder()
for col in ["Airline", "Origin", "Destination", "Weather", "Aircraft_Type", "Holiday"]:
    df[col] = le.fit_transform(df[col])

print("Categorical columns encoded")

# Remove unnecessary columns

df = df.drop(["Flight_ID", "Date"], axis=1)
print("Dataset shape:", df.shape)

# Remove leakage and identifier features

df = df.drop(["Departure_Delay_Min", "Flight_Number"], axis=1)

print("Columns:", df.columns.tolist())
df.head()

# Separate features and target

X = df.drop("Delay_Status", axis=1)
y = df["Delay_Status"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Target Class Distribution

df["Delay_Status"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", startangle=90
)
plt.title("Flight Delay Status Distribution")
plt.ylabel("")
plt.show()

# Check Class Balance

print(df["Delay_Status"].value_counts())
print(df["Delay_Status"].value_counts(normalize=True).mul(100).round(2))

# Check outliers

for col in ["Temperature_C", "Wind_Speed_kmph", "Distance_km"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    print(col, "Outliers:", ((df[col] < q1-1.5*(q3-q1)) | (df[col] > q3+1.5*(q3-q1))).sum())

# Split data into training and testing data

X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("70-30:", X_train_70.shape, X_test_70.shape)
print("80-20:", X_train_80.shape, X_test_80.shape)

# Scale the features

scaler_70 = StandardScaler()
X_train_70 = scaler_70.fit_transform(X_train_70)
X_test_70 = scaler_70.transform(X_test_70)

scaler_80 = StandardScaler()
X_train_80 = scaler_80.fit_transform(X_train_80)
X_test_80 = scaler_80.transform(X_test_80)

print("Feature scaling completed")

# Logistic Regression

lr_70 = LogisticRegression(max_iter=10000)
lr_70.fit(X_train_70, y_train_70)
y_pred_lr_70 = lr_70.predict(X_test_70)

lr_80 = LogisticRegression(max_iter=10000)
lr_80.fit(X_train_80, y_train_80)
y_pred_lr_80 = lr_80.predict(X_test_80)

print("70-30 Accuracy:", accuracy_score(y_test_70, y_pred_lr_70))
print(classification_report(y_test_70, y_pred_lr_70))

print("80-20 Accuracy:", accuracy_score(y_test_80, y_pred_lr_80))
print(classification_report(y_test_80, y_pred_lr_80))

# KNN

knn_70 = KNeighborsClassifier(n_neighbors=5)
knn_70.fit(X_train_70, y_train_70)
y_pred_knn_70 = knn_70.predict(X_test_70)

knn_80 = KNeighborsClassifier(n_neighbors=5)
knn_80.fit(X_train_80, y_train_80)
y_pred_knn_80 = knn_80.predict(X_test_80)

print("70-30 Accuracy:", accuracy_score(y_test_70, y_pred_knn_70))
print(classification_report(y_test_70, y_pred_knn_70))

print("80-20 Accuracy:", accuracy_score(y_test_80, y_pred_knn_80))
print(classification_report(y_test_80, y_pred_knn_80))

# Decision Tree

dt_70 = DecisionTreeClassifier(random_state=42)
dt_70.fit(X_train_70, y_train_70)
y_pred_dt_70 = dt_70.predict(X_test_70)

dt_80 = DecisionTreeClassifier(random_state=42)
dt_80.fit(X_train_80, y_train_80)
y_pred_dt_80 = dt_80.predict(X_test_80)

print("70-30 Accuracy:", accuracy_score(y_test_70, y_pred_dt_70))
print(classification_report(y_test_70, y_pred_dt_70))

print("80-20 Accuracy:", accuracy_score(y_test_80, y_pred_dt_80))
print(classification_report(y_test_80, y_pred_dt_80))

# Naive Bayes

nb_70 = GaussianNB()
nb_70.fit(X_train_70, y_train_70)
y_pred_nb_70 = nb_70.predict(X_test_70)

nb_80 = GaussianNB()
nb_80.fit(X_train_80, y_train_80)
y_pred_nb_80 = nb_80.predict(X_test_80)

print("70-30 Accuracy:", accuracy_score(y_test_70, y_pred_nb_70))
print(classification_report(y_test_70, y_pred_nb_70))

print("80-20 Accuracy:", accuracy_score(y_test_80, y_pred_nb_80))
print(classification_report(y_test_80, y_pred_nb_80))

# Random Forest

rf_70 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_70.fit(X_train_70, y_train_70)
y_pred_rf_70 = rf_70.predict(X_test_70)

rf_80 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_80.fit(X_train_80, y_train_80)
y_pred_rf_80 = rf_80.predict(X_test_80)

print("70-30 Accuracy:", accuracy_score(y_test_70, y_pred_rf_70))
print(classification_report(y_test_70, y_pred_rf_70))

print("80-20 Accuracy:", accuracy_score(y_test_80, y_pred_rf_80))
print(classification_report(y_test_80, y_pred_rf_80))

# Linear SVM

svm_80 = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=2000
)

svm_80.fit(X_train_80, y_train_80)
y_pred_svm_80 = svm_80.predict(X_test_80)

print("SVM Accuracy:", round(accuracy_score(y_test_80, y_pred_svm_80), 4))
print(classification_report(y_test_80, y_pred_svm_80))

# Confusion Matrix Visualization

cm = confusion_matrix(y_test_80, y_pred_rf_80)

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix - Random Forest (80-20)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["No Delay", "Delay"])
plt.yticks([0, 1], ["No Delay", "Delay"])

# Display values inside the matrix

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.colorbar()

plt.tight_layout()
plt.show()

# Compare model accuracies

results = {
    "Logistic Regression": [
        accuracy_score(y_test_70, y_pred_lr_70),
        accuracy_score(y_test_80, y_pred_lr_80)    ],
    "KNN": [
        accuracy_score(y_test_70, y_pred_knn_70),
        accuracy_score(y_test_80, y_pred_knn_80)    ],
    "Decision Tree": [
        accuracy_score(y_test_70, y_pred_dt_70),
        accuracy_score(y_test_80, y_pred_dt_80)    ],
    "Naive Bayes": [
        accuracy_score(y_test_70, y_pred_nb_70),
        accuracy_score(y_test_80, y_pred_nb_80)    ],
    "Random Forest": [
        accuracy_score(y_test_70, y_pred_rf_70),
        accuracy_score(y_test_80, y_pred_rf_80)    ],
   "SVM": [
    None,
    accuracy_score(y_test_80, y_pred_svm_80)
]
}

comparison = pd.DataFrame(results, index=["70-30", "80-20"]).T
comparison.columns = ["70-30 Accuracy", "80-20 Accuracy"]

print(comparison)

# Model Training Time

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train_80, y_train_80)
    print(name, "Time:", round(time.time() - start, 3), "sec")

# 5-Fold Cross Validation

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_80, y_train_80, cv=5)
    print(name, "Mean:", round(scores.mean(), 4),
          "Std:", round(scores.std(), 4))

# Grid Search for all models

param_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=10000),
        {"C": [0.1, 1, 10]}
    ),

    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7]}
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [10, 20, None]}
    ),

    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-9, 1e-8, 1e-7]}
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [10, 20]}
    ),

    "SVM": (
        LinearSVC(class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10]}
    )
}

best_models = {}

for name, (model, params) in param_grids.items():

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )

    grid.fit(X_train_80, y_train_80)

    best_models[name] = grid.best_estimator_

    print(name)
    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))
    print()

# Tuned Model Results

tuned_results = {}
for name, model in best_models.items():
      y_pred = model.predict(X_test_80)
      accuracy = accuracy_score(y_test_80, y_pred)
      tuned_results[name] = accuracy
      print(name, "Accuracy:", round(accuracy, 4))

# Tuned Model Accuracy Comparison

models_name = list(tuned_results.keys())
accuracies = list(tuned_results.values())

plt.figure(figsize=(10, 5))
plt.bar(models_name, accuracies)

plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy")
plt.title("Tuned Model Accuracy Comparison")

plt.xticks(rotation=30)
plt.ylim(0.60, 0.90)

plt.tight_layout()
plt.show()

# Confusion Matrix for the Best Model

best_name = max(tuned_results, key=tuned_results.get)
best_model = best_models[best_name]
y_pred_best = best_model.predict(X_test_80)

cm = confusion_matrix(y_test_80, y_pred_best)

print("Best Model:", best_name)
print("Confusion Matrix:")
print(cm)

# Classification Report for the Best Model

print("Best Model:", best_name)
print(classification_report(y_test_80, y_pred_best))

# Final Model Comparison

final_results = pd.DataFrame({
    "Model": list(tuned_results.keys()),
    "Accuracy": list(tuned_results.values())
}).sort_values("Accuracy", ascending=False)

print(final_results)

# Best Model Result

print("Best Model:", best_name)
print("Train-Test Split: 80-20")
print("Accuracy:", round(tuned_results[best_name], 4))

# Final Model Comparison

final_results = pd.DataFrame({
    "Model": list(tuned_results.keys()),
    "Accuracy": list(tuned_results.values())
}).sort_values("Accuracy", ascending=False)

print(final_results)

# Final Best Model

print("Best Model:", best_name)
print("Accuracy:", round(tuned_results[best_name], 4))

# Final Model Accuracy Graph

plt.figure(figsize=(10, 5))

plt.bar(final_results["Model"], final_results["Accuracy"])

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Final Model Accuracy Comparison")

plt.xticks(rotation=30)
plt.ylim(0.60, 0.90)

plt.tight_layout()
plt.show()
