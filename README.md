# Graduate Admission Prediction using Neural Networks

##  Project Overview

This project builds and deploys a **neural network regression model** to predict the **Chance of Admission** for graduate students.
It has two components:

1. **Model Training (`train_model.py`)** – trains a neural network on the dataset and saves the model + scaler.
2. **Prediction App (`predict_app.py`)** – a simple GUI (Tkinter) where users enter their scores and get admission chance predictions in real time.

---

##  Dataset

* **File:** `Admission_Predict_Ver1.1.csv`
* **Source:** Graduate Admission dataset (Kaggle / UCI ML repository).
* **Target Variable:** `Chance of Admit`
* **Features:**

  * GRE Score
  * TOEFL Score
  * University Rating
  * SOP (Statement of Purpose strength)
  * LOR (Letter of Recommendation strength)
  * CGPA (Cumulative GPA)
  * Research (1 if research experience, 0 otherwise)

---

##  Steps in Training (`train_model.py`)

1. Load and clean dataset (drop irrelevant columns).
2. Preprocess:

   * Split train/test
   * Normalize features using `MinMaxScaler`
3. Build Neural Network:

   * Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Linear)
4. Compile with:

   * Loss = Mean Squared Error (MSE)
   * Optimizer = Adam
   * Metric = Mean Absolute Error (MAE)
5. Train for **50 epochs** with validation split.
6. Save:

   * Trained model (`admission_model.keras`)
   * Scaler (`scaler.pkl`)
7. Plot training curves (Loss & MAE).

---

## 🖥 Prediction App (`predict_tk.py`)

* **Framework:** Tkinter (Python built-in GUI).
* **Inputs:** GRE, TOEFL, Rating, SOP, LOR, CGPA, Research.
* **Outputs:** Predicted admission probability (0–1).
* **Extra:** Displays status message (High / Moderate / Low chance).

---

##  Results

* Model achieves reasonable prediction accuracy (measured by MAE).
* Training and validation curves show learning progress.
* GUI allows easy “what-if” analysis of admission chances.

---

##  How to Run

### 1. Train the Model

```bash
python train_model.py
```

This will generate:

* `admission_model.keras` (trained model)
* `scaler.pkl` (normalization scaler)

### 2. Run the GUI

```bash
python predict_tk.py
```

Enter your scores in the form, click **Predict**, and see your estimated chance of admission.

---

##  Requirements

Install required libraries:

```bash
pip install pandas scikit-learn tensorflow matplotlib joblib
```

Tkinter comes pre-installed with Python (no extra install needed).

---

##  Notes

* If you already have a model in `.h5` format, it will still work — but `.keras` is recommended for Keras 3.x+.
* Make sure `admission_model.keras` (or `.h5`) and `scaler.pkl` are in the same directory as `predict_tk.py`.
