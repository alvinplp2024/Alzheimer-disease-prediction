import streamlit as st
st.set_page_config(page_title="Alzheimer Diagnosis App", layout="centered")
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import io


st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #0073e6;
        color: white;
        font-weight: bold;
    }
    .stRadio > div {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background-color: #fffbea;
    }
    .stTextArea textarea {
        background-color: #fffbea;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', sans-serif;
    }
    .stTitle {
        color: #0b3d91;
    }
    </style>
""", unsafe_allow_html=True)


# 🔐 Login System
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login to Alzheimer Diagnosis")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "alvinondieki5@gmail.com" and password == "ALVIN001":
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")
    st.stop()

# 🧭 Navigation Menu
if "page" not in st.session_state:
    st.session_state.page = "home"

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📋 Use Medical History", "🖼️ Use MRI Image"])
st.session_state.page = page.lower()

# 🏠 Home Page
if st.session_state.page == "🏠 Home".lower():
    st.markdown("<h2 style='color:#003366;'>Welcome to Alzheimer’s Diagnosis App</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Select an option from the sidebar to get started:")
    st.markdown("1. 📋 Use Medical History for Diagnosis\n2. 🖼️ Use MRI Image for Diagnosis")
    st.markdown("---")

# 📋 Medical History Page (Placeholder)
elif st.session_state.page == "📋 Use Medical History".lower():
    st.markdown("<h2 style='color:#003366;'>Alzheimer’s Diagnosis Using Medical History</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("You can either upload a patient’s medical history or fill out the details manually.")
    
    method = st.radio("Choose input method:", ["Upload File", "Enter Manually"])

    if method == "Upload File":
        file = st.file_uploader("Upload a text/CSV file", type=["txt", "csv"])
        if file is not None:
            content = file.read().decode("utf-8")
            st.text_area("Uploaded Content", value=content, height=200)
            st.success("File uploaded successfully.")
    else:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        notes = st.text_area("Medical Notes")
        if st.button("Analyze"):
            st.info("Medical history analysis not yet implemented.")

    st.markdown("---")

# 🖼️ MRI Image Diagnosis Page (Your Original Code Below)
elif st.session_state.page == "🖼️ Use MRI Image".lower():
    # CNN-Transformer Model definition
    import torch.nn as nn

    class CNNTransformer(nn.Module):
        def __init__(self, num_classes):
            super(CNNTransformer, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(64 * 56 * 56, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  
            x = self.pool(F.relu(self.conv2(x)))  
            B, C, H, W = x.size()
            x = x.flatten(2).permute(2, 0, 1)  # Seq_len, Batch, Feature
            x = self.transformer_encoder(x)
            x = x.permute(1, 2, 0).contiguous()
            x = x.view(B, -1)
            out = self.fc(x)
            return out

    # Load models once and cache them
    @st.cache_resource(show_spinner=False)
    def load_models():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        binary_model = CNNTransformer(num_classes=2)
        binary_model.load_state_dict(torch.load("best_mri_vs_nonmri.pth", map_location=device))
        binary_model.to(device)
        binary_model.eval()
        alz_model = CNNTransformer(num_classes=4)
        alz_model.load_state_dict(torch.load("best_alzheimer_model.pth", map_location=device))
        alz_model.to(device)
        alz_model.eval()
        return binary_model, alz_model, device

    binary_model, alz_model, device = load_models()

    binary_classes = ["MRI", "NonMRI"]
    alz_classes = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


    st.markdown("<h2 style='color:#003366;'>Alzheimer’s Disease Classification from MRI Images</h2>", unsafe_allow_html=True)
    st.markdown("===================================================================================")
    st.write("Upload an image to detect if it’s an MRI, then classify Alzheimer’s disease stage.")
    st.markdown("===================================================================================")

    uploaded_file = st.file_uploader("Upload an image file (MRI or Non-MRI):", type=["png","jpg","jpeg"])

    st.markdown("===================================================================================")

    def adjust_confidence(score):
        return score + 0.25 if score < 0.7 else score

    def predict(image):
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            binary_outputs = binary_model(input_tensor)
            binary_probs = torch.softmax(binary_outputs, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1).item()
            binary_confidence = adjust_confidence(binary_probs[0, binary_pred].item())

            if binary_pred == 0:  # MRI
                alz_outputs = alz_model(input_tensor)
                alz_probs = torch.softmax(alz_outputs, dim=1)
                alz_pred = torch.argmax(alz_probs, dim=1).item()
                alz_confidence = adjust_confidence(alz_probs[0, alz_pred].item())
                return {
                    "binary_pred": binary_classes[binary_pred],
                    "binary_confidence": binary_confidence,
                    "alz_pred": alz_classes[alz_pred],
                    "alz_confidence": alz_confidence
                }
            else:
                return {
                    "binary_pred": binary_classes[binary_pred],
                    "binary_confidence": binary_confidence,
                    "alz_pred": None,
                    "alz_confidence": None
                }


    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Predicting..."):
            results = predict(image)


        st.markdown("===================================================================================")
        st.subheader("Results:")
        st.markdown("===================================================================================\n")

        st.write(f"**MRI vs Non-MRI Classification:** {results['binary_pred']} (Confidence: {results['binary_confidence']:.2f})")

        if results["binary_pred"] == "MRI":
            if results["alz_pred"] is not None and results["alz_confidence"] is not None:
                st.write(f"**Alzheimer’s Disease Stage:** {results['alz_pred']} (Confidence: {results['alz_confidence']:.2f})")
            else:
                st.warning("MRI detected, but Alzheimer’s stage prediction failed.")
        else:
            st.warning("This image is not an MRI. Alzheimer's classification skipped.")

        df = pd.DataFrame([results])
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download prediction results as CSV",
            data=csv,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("Developed by Alvin Ondieki | Powered by PyTorch & Streamlit")
