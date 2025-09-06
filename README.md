# Graduate Admission Prediction using Neural Networks

## 📌 Project Overview

This project builds a **regression model using a neural network** to predict the **Chance of Admission** for graduate students.
The dataset includes features such as **GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research Experience**.

The goal is to train a model that can predict the probability of admission based on these factors.

---

## Dataset

* **File:** `Admission_Predict_Ver1.1.csv`
* **Source:** Graduate Admission dataset (commonly available on Kaggle / UCI ML repository).
* **Target variable:** `Chance of Admit`
* **Features:**

  * GRE Score
  * TOEFL Score
  * University Rating
  * SOP (Statement of Purpose strength)
  * LOR (Letter of Recommendation strength)
  * CGPA (Cumulative GPA)
  * Research (1 if research experience, 0 otherwise)

---

##  Steps in the Code

1. **Load and inspect dataset**

   * Read CSV file
   * Check for missing values
   * Drop irrelevant column (`Serial No.`)

2. **Preprocessing**

   * Separate features (`X`) and target (`y`)
   * Train-test split (80% train, 20% test)
   * Normalize features using `MinMaxScaler`

3. **Model Architecture**

   * Input layer: matches number of features
   * Hidden layers:

     * Dense(64, ReLU)
     * Dense(32, ReLU)
   * Output layer: Dense(1, Linear) → regression output

4. **Compilation**

   * Loss: Mean Squared Error (MSE)
   * Optimizer: Adam
   * Metric: Mean Absolute Error (MAE)

5. **Training**

   * 50 epochs
   * 20% of training data used for validation

6. **Evaluation**

   * Test set evaluation with MAE
   * Visualization of training/validation loss & MAE

---

##  Results

* The model outputs **Mean Absolute Error (MAE)** on test data.
* Training and validation curves are plotted to visualize model performance.

---

##  How to Run

1. Clone the repository / copy the script.
2. Place the dataset file (`Admission_Predict_Ver1.1.csv`) in the same directory.
3. Install dependencies:

   ```bash
   pip install pandas scikit-learn tensorflow matplotlib
   ```
4. Run the script:

   ```bash
   python admission_prediction.py
   ```
