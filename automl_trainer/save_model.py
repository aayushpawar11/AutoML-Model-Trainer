import joblib

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"[INFO] Model saved as {filename}")
