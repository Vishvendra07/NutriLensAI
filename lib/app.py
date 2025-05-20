import streamlit as st
import easyocr
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import joblib

# --- Updated MLP Model (Improved Accuracy) ---
class RuleAugmentedMLP(nn.Module):
    def __init__(self, input_dim):
        super(RuleAugmentedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(F.leaky_relu(self.fc3(x)))
        return self.output(x)

# --- OCR Setup ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Nutrition Parsing ---
def extract_nutrition_info(text_lines):
    nutrients = {
        'calories': 0, 'fat': 0, 'saturated_fat': 0, 'sodium': 0,
        'sugar': 0, 'fiber': 0, 'protein': 0
    }
    for line in text_lines:
        line = line.lower()
        if 'calories' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['calories'] = int(match.group(1))
        elif 'total fat' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['fat'] = int(match.group(1))
        elif 'saturated fat' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['saturated_fat'] = int(match.group(1))
        elif 'sodium' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['sodium'] = int(match.group(1))
        elif 'sugars' in line or 'sugar' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['sugar'] = int(match.group(1))
        elif 'fiber' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['fiber'] = int(match.group(1))
        elif 'protein' in line:
            match = re.search(r'(\d+)', line)
            if match: nutrients['protein'] = int(match.group(1))
    return nutrients

def rule_based_tags(nutrients, concern):
    tags = []
    if nutrients['sodium'] > 400:
        tags.append("High Sodium")
    if nutrients['sugar'] > 10 and concern == "Diabetes":
        tags.append("High Sugar for Diabetes")
    elif nutrients['sugar'] > 20:
        tags.append("High Sugar")
    if nutrients['saturated_fat'] > 5:
        tags.append("High Saturated Fat")
    if nutrients['fiber'] >= 5:
        tags.append("Good Fiber")
    return tags

def encode_features(nutrients, tags):
    features = list(nutrients.values())
    rules = [
        1 if "High Sodium" in tags else 0,
        1 if "High Sugar" in tags or "High Sugar for Diabetes" in tags else 0,
        1 if "High Saturated Fat" in tags else 0,
        1 if "Good Fiber" in tags else 0,
    ]
    return np.array(features + rules, dtype=np.float32)

def generate_explanation(tags, concern):
    reasons = []
    if not tags:
        reasons.append("✅ This item met all health criteria. No concerns were flagged based on your profile.")
    else:
        for tag in tags:
            if tag == "High Sodium":
                reasons.append("⚠️ High sodium may increase blood pressure.")
            if tag == "High Sugar" or tag == "High Sugar for Diabetes":
                reasons.append("⚠️ High sugar content is a risk for blood sugar spikes.")
            if tag == "High Saturated Fat":
                reasons.append("⚠️ Saturated fat may raise LDL cholesterol.")
            if tag == "Good Fiber":
                reasons.append("✅ Good fiber content supports digestion.")
    return reasons

def radar_chart(nutrients):
    labels = list(nutrients.keys())
    values = [v if v > 0 else 0.5 for v in nutrients.values()]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    st.pyplot(fig)

def bar_chart(nutrients):
    display_values = {k: (v if v > 0 else "Not Found") for k, v in nutrients.items()}
    fig, ax = plt.subplots()
    ax.barh(list(display_values.keys()), [v if isinstance(v, int) else 0 for v in display_values.values()], color='lightgreen')
    ax.set_xlabel("Grams / mg")
    ax.set_title("Nutrient Composition")
    st.pyplot(fig)

# --- Load Model + Scaler ---
model_path = '/content/drive/MyDrive/AI/best_model.pth'
scaler_path = '/content/drive/MyDrive/AI/scaler.pkl'

model = RuleAugmentedMLP(input_dim=11)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load(scaler_path)

# --- Streamlit Config ---
st.set_page_config(page_title="NutriLensAI", layout="centered")

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🥦 NutriLensAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Smart Nutrition Label Analyzer powered by AI + Rules</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Upload & Process Flow ---
st.markdown("### 📤 Upload a Nutrition Label")
uploaded_file = st.file_uploader("Upload JPG or PNG file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img.thumbnail((300, 300))
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create two columns
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.image(img, caption="📋 Uploaded Label", width=200)

    with right_col:
        st.markdown("### 👤 Select Your Health Concern")
        user_concern = st.selectbox("Choose one:", 
            ["None", "Heart Health", "Diabetes", "Weight Loss", "General Fitness"])

    # --- OCR + Feature Extraction ---
    text = reader.readtext("temp.jpg", detail=0)
    nutrients = extract_nutrition_info(text)
    tags = rule_based_tags(nutrients, user_concern)
    input_vector = encode_features(nutrients, tags)

    # --- Normalize Input ---
    input_vector_scaled = scaler.transform([input_vector])  # 2D array
    tensor_input = torch.tensor(input_vector_scaled, dtype=torch.float32)

    # --- Charts ---
    st.markdown("### 📊 Nutritional Composition")
    col1, col2 = st.columns(2)
    with col1:
        bar_chart(nutrients)
    with col2:
        radar_chart(nutrients)

    # --- Rule-based Warnings ---
    st.markdown("### 🚦 Rule-Based Warnings")
    if tags:
        for tag in tags:
            st.warning(f"⚠️ {tag}")
    else:
        st.success("✅ No rule-based warnings triggered.")

    # --- Model Inference ---
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        prediction = np.argmax(probs)
        classes = ["Healthy", "Caution", "Avoid"]
        final_class = classes[prediction]

    st.markdown("### 🧠 Final Classification")
    color_map = {"Healthy": "green", "Caution": "orange", "Avoid": "red"}
    st.markdown(f"**Result:** <span style='color:{color_map[final_class]}; font-weight:bold'>{final_class}</span>", unsafe_allow_html=True)
    st.progress(float(probs[prediction]))
    st.caption(f"Confidence: {probs[prediction]*100:.2f}%")

    # --- Explanation ---
    st.markdown("### 💡 Why this classification?")
    explanation = generate_explanation(tags, user_concern)
    for line in explanation:
        if line.startswith("⚠️"):
            st.error(line)
        elif line.startswith("✅"):
            st.success(line)

    st.markdown("---")
    st.info("💡 Tip: Try uploading labels from your favorite snacks, beverages, or frozen meals to compare results.")

else:
    st.warning("👈 Upload a nutrition label to begin analysis.")
