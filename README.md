# 🖼️ AI Image Caption Generator

A Streamlit web app that generates captions for images using a vision-language model and allows AI suggestions via GPT-2.

## 🚀 Features
- Upload an image and get multiple AI-generated captions.
- Ask the AI to improve captions (funny, poetic, etc.).
- Download captions as TXT or JSON.

## 🧰 Technologies Used
- Python, Streamlit
- Hugging Face Transformers
- PyTorch
- Pillow
- Streamlit Chat

## 📸 Demo Screenshots

### Upload Interface
![Upload Screenshot](https://github.com/pranathi625/AI_Image_Caption_Generator/blob/main/assets/Screenshot%202025-09-30%20190725.png?raw=true)

### Caption Output
![Caption Output](https://github.com/pranathi625/AI_Image_Caption_Generator/blob/main/assets/Screenshot%202025-09-30%20190655.png?raw=true)

### Full Interface
![Full Interface](https://github.com/pranathi625/AI_Image_Caption_Generator/blob/main/assets/Screenshot%202025-09-30%20190314.png?raw=true)

## 🛠️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/pranathi625/AI_Image_Caption_Generator.git
cd AI_Image_Caption_Generator

2. Set Up Environment & Install Dependencies

# Create virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run app.py


4. Open in Browser
After running the above command, Streamlit will provide a local URL like:
Local URL: http://localhost:8501
Network URL: http://192.168.xx.xx:8501
