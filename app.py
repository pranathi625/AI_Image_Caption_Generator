import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from streamlit_chat import message  # pip install streamlit-chat
import numpy as np

# -----------------------------
# Helper: Prepare any type of image
# -----------------------------
def prepare_image(image):
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except Exception:
            return None

    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 3:
            image = image[:, :, ::-1]
        image = Image.fromarray(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

# -----------------------------
# Load Image Captioning Model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_caption_model(model_name="nlpconnect/vit-gpt2-image-captioning"):
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

caption_model, feature_extractor, tokenizer, device = load_caption_model()

# -----------------------------
# Load GPT-2 Text Model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_text_model(model_name="gpt2-medium"):
    text_tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = AutoModelForCausalLM.from_pretrained(model_name)
    text_pipe = pipeline(
        "text-generation",
        model=text_model,
        tokenizer=text_tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return text_pipe

text_generator = load_text_model()

# -----------------------------
# Generate Captions
# -----------------------------
def generate_captions(image, max_length=50, num_beams=4, num_return_sequences=3,
                      do_sample=True, top_k=50, top_p=0.95, temperature=1.0):
    if num_beams < num_return_sequences:
        num_beams = num_return_sequences

    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    outputs = caption_model.generate(
        pixel_values,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    return captions

# -----------------------------
# Generate AI Chat Response
# -----------------------------
def get_ai_response(user_input, last_caption):
    if not user_input.strip():
        return ""
    
    prompt = (
        f"Original: {last_caption}\n"
        f"Poetic:"
    )
    try:
        response = text_generator(
            prompt,
            max_length=60,
            do_sample=True,
            top_p=0.95,
            temperature=1.2,
            num_return_sequences=1
        )
        poetic_caption = response[0]["generated_text"].replace(prompt, "").strip()
        return poetic_caption
    except Exception:
        return "Sorry, I couldn't generate a suggestion."

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Image Caption Generator", layout="wide")
st.markdown("<h1 style='text-align: center;'>🖼️ AI Image Caption Generator</h1>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("Caption Settings")
max_length = st.sidebar.slider("Max Caption Length", 10, 100, 50)
num_beams = st.sidebar.slider("Beam Search Width", 1, 10, 4)
num_return_sequences = st.sidebar.slider("Number of Captions", 1, 5, 3)
do_sample = st.sidebar.checkbox("Use Sampling", True)
top_k = st.sidebar.slider("Top-k (Sampling)", 10, 100, 50)
top_p = st.sidebar.slider("Top-p (Sampling)", 0.1, 1.0, 0.95)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)

# Initialize session state
if "captions_list" not in st.session_state:
    st.session_state.captions_list = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Layout
col1, col2 = st.columns([2, 1])

# --- Left: Image & Captions ---
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            image = prepare_image(Image.open(uploaded_file))
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            image = None
        if image is not None:
            # Buttons for generating/regenerating captions
            if not st.session_state.captions_list:
                if st.button("Generate Caption"):
                    with st.spinner("Generating captions..."):
                        st.session_state.captions_list = generate_captions(
                            image, max_length, num_beams, num_return_sequences,
                            do_sample, top_k, top_p, temperature
                        )
            else:
                if st.button("Regenerate Caption"):
                    with st.spinner("Regenerating captions..."):
                        st.session_state.captions_list = generate_captions(
                            image, max_length, num_beams, num_return_sequences,
                            do_sample, top_k, top_p, temperature
                        )

        # Show captions
        if st.session_state.captions_list:
            st.markdown("### Generated Captions")
            for idx, cap in enumerate(st.session_state.captions_list):
                st.markdown(f"**Caption {idx+1}:** {cap}")

            col_txt, col_json = st.columns(2)
            with col_txt:
                st.download_button(
                    "📄 Download TXT",
                    "\n".join(st.session_state.captions_list),
                    file_name="captions.txt"
                )
            with col_json:
                st.download_button(
                    "📦 Download JSON",
                    json.dumps(st.session_state.captions_list, indent=2),
                    file_name="captions.json"
                )

# --- Right: Chatbox ---
with col2:
    st.markdown("### Chat with AI about your image")

chat_container = st.container()

with st.form(key="chat_form"):
    user_input = st.text_input("Enter your suggestion:", key="chat_input_field")
    submit_btn = st.form_submit_button("Send")

    if submit_btn and user_input.strip():
        last_caption = st.session_state.captions_list[0] if st.session_state.captions_list else "No caption yet."
        ai_response = get_ai_response(user_input, last_caption)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", f"Suggested Caption: {ai_response}"))

# Display chat
with chat_container:
    for speaker, text in st.session_state.chat_history:
        message(text, is_user=(speaker == "You"))
