**Project Overview**

The Medic Handwriting to Text Conversion project is a Streamlit-based web application designed to convert handwritten medical prescriptions into structured digital text. It leverages deep learning models, including OpenAI's language models and computer vision techniques, to accurately recognize and parse handwritten text.

**Project Workflow**

Input Data Handling: Users upload images of handwritten prescriptions.

Preprocessing: Image enhancement techniques like grayscale conversion and noise reduction are applied.

Text Extraction: Handwritten text is extracted using a trained model.

Language Model Parsing: The extracted text is passed through OpenAI's language model for contextual understanding.

Output Generation: The final parsed text is displayed and can be exported as a structured digital file.
**
Key Features
**
Handwriting recognition for medical prescriptions.

Streamlit-based user-friendly interface.

Support for multiple languages.

Secure and private handling of sensitive medical data.

**System Requirements**

Python 3.10 or higher

Streamlit

OpenAI API

Langchain

Pillow

Pandas

NumPy

**Installation Steps**

Clone the Repository

git clone https://github.com/your-username/medic_handwriting_to_text_conv.git

cd medic_handwriting_to_text_conv

Create a Virtual Environment

python -m venv venv

Activate the Virtual Environment

**For Windows:**

venv\Scripts\activate
**
For macOS/Linux:**

source venv/bin/activate

**Install the Dependencies**

pip install -r requirements.txt

**Set Up API Key for OpenAI**

Create a .streamlit/secrets.toml file.

[secrets]
OPENAI_API_KEY = "your_openai_api_key"

**Run the Streamlit App**

streamlit run prescription.py

**Execution Flow
**
The user uploads an image of a handwritten prescription.

The image undergoes preprocessing for noise reduction and enhancement.

The text extraction model extracts text from the image.

The extracted text is sent to OpenAI's model for language understanding.

The final output is displayed on the Streamlit interface.
**
Outputs
**
Extracted text in a structured format.

Downloadable text or CSV file.

**Troubleshooting**

Streamlit Page Config Error: Ensure st.set_page_config() is the first Streamlit command.

Module Not Found Error: Install missing modules with pip install module_name.

API Key Error: Verify the correct path and content of the .streamlit/secrets.toml file.

**Future Scope**

Multilingual handwriting recognition.

Integration with Electronic Health Records (EHR) systems.

Improved accuracy for complex prescriptions.

**Contributors**

Aleena Rose K Sunil
Anna Biju
Dora Liz Dileep
Aravind Saju Krishna

