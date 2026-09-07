# ✈️ Flight Delay Prediction Using Machine Learning

A Machine Learning project developed to predict whether a flight will be **Delayed** or **Not Delayed** using flight-related and environmental information.

## 📌 Project Overview

Flight delays are a common issue that can affect passengers, airlines, and airport operations. This project uses Machine Learning classification algorithms to analyze different flight and environmental features and predict the delay status of a flight.

Multiple Machine Learning models were implemented, evaluated, compared, and tuned to identify the best-performing model.

## 🎯 Objectives

* Predict whether a flight will be delayed or not.
* Clean and preprocess the flight dataset.
* Apply different Machine Learning classification algorithms.
* Compare the performance of different models.
* Perform 5-Fold Cross-Validation.
* Apply GridSearchCV for hyperparameter tuning.
* Select the best-performing model for flight delay prediction.

## 📊 Dataset

**Dataset:** `Flight_Delay_Prediction_Dataset_Raw.csv`

* Initial records: **65,000**
* Records after duplicate removal: **63,729**
* Final records after preprocessing and outlier removal: **63,094**
* Number of input features: **10**
* Target variable: **Delay_Status**

### Features Used

| Feature         | Description                                  |
| --------------- | -------------------------------------------- |
| Airline         | Airline operating the flight                 |
| Origin          | Flight departure location                    |
| Destination     | Flight arrival location                      |
| Departure_Hour  | Scheduled departure hour                     |
| Weather         | Weather condition                            |
| Temperature_C   | Temperature in Celsius                       |
| Wind_Speed_kmph | Wind speed in km/h                           |
| Distance_km     | Flight distance in kilometres                |
| Aircraft_Type   | Type of aircraft                             |
| Holiday         | Indicates whether the flight is on a holiday |

### Target Variable

**Delay_Status**

* `0` → No Delay
* `1` → Delay

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Removed outliers using the IQR method.
3. Handled missing numerical values using the median.
4. Handled missing categorical values using the mode.
5. Encoded categorical features using `LabelEncoder`.
6. Removed unnecessary and leakage-related features:

   * `Flight_ID`
   * `Date`
   * `Flight_Number`
   * `Departure_Delay_Min`
7. Applied `StandardScaler` for feature scaling.
8. Divided the dataset into training and testing sets.

## 🤖 Machine Learning Models

The following classification algorithms were implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Gaussian Naive Bayes
5. Random Forest
6. Linear Support Vector Machine (SVM)

## 📈 Model Performance

### Accuracy Before Tuning

| Model               | 70–30 Split | 80–20 Split |
| ------------------- | ----------: | ----------: |
| Logistic Regression |      73.64% |      73.58% |
| KNN                 |      76.81% |      77.01% |
| Decision Tree       |      80.26% |      79.57% |
| Naive Bayes         |      76.11% |      76.17% |
| Random Forest       |  **84.92%** |  **84.95%** |
| Linear SVM          |           — |      65.80% |

### Tuned Model Accuracy

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression |     73.60% |
| KNN                 |     77.47% |
| Decision Tree       |     84.47% |
| Naive Bayes         |     76.17% |
| Random Forest       | **84.76%** |
| Linear SVM          |     65.79% |

## 🏆 Best Model

**Random Forest** was selected as the final model because it achieved the best overall performance.

### Final Results

* **Train-Test Split:** 80–20
* **Test Accuracy:** 84.76%
* **Cross-Validation Accuracy:** 84.82%
* **Precision for Delay:** 0.89
* **Recall for Delay:** 0.89
* **F1-Score for Delay:** 0.89

### Best Hyperparameters

```text
n_estimators = 100
max_depth = 20
min_samples_split = 5
```

## 📊 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* 5-Fold Cross-Validation
* GridSearchCV

## ⏱️ Model Training Time

| Model               | Training Time |
| ------------------- | ------------: |
| Logistic Regression |     0.062 sec |
| KNN                 |     0.469 sec |
| Decision Tree       |     0.437 sec |
| Naive Bayes         |     0.047 sec |
| Random Forest       |    13.576 sec |
| Linear SVM          |     0.109 sec |

## 📁 Project Structure

```text
Flight-Delay-Prediction-ML/
│
├── Doc/
│   └── Project Documentation
│
├── Coding/
│   ├── Dataset
│   └── MainFile.py
│
└── README.md
```

### Folder Description

* **Doc/** – Contains the project documentation.
* **Coding/** – Contains the project source code and database.
* **README.md** – Contains the project overview, methodology, technologies, and results.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **GridSearchCV**

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/bharadkashyap/Flight-Delay-Prediction-ML.git
```

### 2. Open the Project

```bash
cd Flight-Delay-Prediction-ML
```

### 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 4. Run the Project

Open the Python file from the **Coding** folder and run.

## 🔍 Conclusion

This project demonstrates the application of Machine Learning for predicting flight delays using flight-related and environmental features. Six classification algorithms were implemented and compared using different evaluation techniques.

After cross-validation and hyperparameter tuning, **Random Forest** achieved the best overall performance with **84.76% test accuracy** and **84.82% cross-validation accuracy**. Therefore, Random Forest was selected as the final model for this project.

The project can be further improved by using larger datasets, real-time weather information, airport traffic data, historical flight information, and advanced Machine Learning techniques.

