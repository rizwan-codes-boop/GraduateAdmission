import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# ---------- Configuration ----------
MODEL_CANDIDATES = ["admission_model.keras", "admission_model.h5"]
SCALER_FILE = "scaler.pkl"
FEATURE_ORDER = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research (0 or 1)"]
# -----------------------------------

# Try to load model (.keras preferred, .h5 fallback)
model = None
for fname in MODEL_CANDIDATES:
    if os.path.exists(fname):
        try:
            model = load_model(fname)
            print(f"Loaded model from: {fname}")
            break
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

if model is None:
    raise FileNotFoundError(
        "No model file found. Put 'admission_model.keras' or 'admission_model.h5' in the app folder."
    )

# Re-compile to attach metrics (silences the TensorFlow warning)
try:
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
except Exception:
    # compilation is optional for inference; ignore failures
    pass

# Load scaler if available
scaler = None
if os.path.exists(SCALER_FILE):
    try:
        scaler = joblib.load(SCALER_FILE)
        print("Loaded scaler:", SCALER_FILE)
    except Exception as e:
        print("Failed to load scaler:", e)

# ---------- GUI ----------
root = tk.Tk()
root.title("Graduate Admission Predictor")
root.geometry("420x420")
root.resizable(False, False)
frm = ttk.Frame(root, padding=12)
frm.pack(fill=tk.BOTH, expand=True)

ttk.Label(frm, text="Enter your details (feature order):").pack(anchor=tk.W)
ttk.Label(frm, text=", ".join(FEATURE_ORDER), wraplength=380).pack(anchor=tk.W, pady=(0,8))

entries = {}

# Create input fields
def make_numeric_row(parent, label_text, default=""):
    row = ttk.Frame(parent)
    row.pack(fill=tk.X, pady=4)
    lbl = ttk.Label(row, text=label_text, width=20, anchor=tk.W)
    lbl.pack(side=tk.LEFT)
    ent = ttk.Entry(row)
    ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return ent

entries["gre"] = make_numeric_row(frm, "GRE Score (0-340)", "300")
entries["toefl"] = make_numeric_row(frm, "TOEFL Score (0-120)", "100")
entries["rating"] = make_numeric_row(frm, "University Rating (1-5)", "3")
entries["sop"] = make_numeric_row(frm, "SOP (1.0-5.0)", "3.0")
entries["lor"] = make_numeric_row(frm, "LOR (1.0-5.0)", "3.0")
entries["cgpa"] = make_numeric_row(frm, "CGPA (0.0-10.0)", "8.0")
entries["research"] = make_numeric_row(frm, "Research (0 or 1)", "0")

result_label = ttk.Label(frm, text="", font=("Segoe UI", 12))
result_label.pack(pady=12)

def predict_action():
    try:
        gre = float(entries["gre"].get())
        toefl = float(entries["toefl"].get())
        rating = float(entries["rating"].get())
        sop = float(entries["sop"].get())
        lor = float(entries["lor"].get())
        cgpa = float(entries["cgpa"].get())
        research = int(float(entries["research"].get()))
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numeric values for all fields.")
        return

    # Basic range checks (non-strict)
    if not (0 <= gre <= 340 and 0 <= toefl <= 120 and 0 <= cgpa <= 10 and research in (0,1)):
        if not messagebox.askyesno("Out of typical range", "Some inputs are outside common ranges. Continue?"):
            return

    features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]], dtype=float)

    if scaler is not None:
        try:
            features = scaler.transform(features)
        except Exception as e:
            messagebox.showwarning("Scaler error", f"Scaler failed to transform: {e}\nUsing raw features.")
    else:
        messagebox.showinfo("No scaler", "No scaler found (scaler.pkl). Using raw feature values directly.")

    try:
        pred = model.predict(features)
        # model.predict returns array shape (1,1) or (1,)
        prediction = float(pred.reshape(-1)[0])
    except Exception as e:
        messagebox.showerror("Prediction error", f"Model prediction failed: {e}")
        return

    # Display
    pct = prediction  # the dataset target is 0..1 (chance)
    result_text = f"Predicted Chance of Admission: {pct:.3f}"
    result_label.config(text=result_text)

    # status message
    if pct > 0.8:
        status = "High chance of admission"
    elif pct > 0.5:
        status = "Moderate chance of admission"
    else:
        status = "Low chance of admission"
    messagebox.showinfo("Result", f"{result_text}*100 \n\n{status}")

btn = ttk.Button(frm, text="Predict", command=predict_action)
btn.pack(pady=6)

# Run
root.mainloop()
