from __future__ import annotations
import base64
import os
import glob
import shutil
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
import streamlit as st
import pandas as pd
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser

# In your prescription.py
import streamlit as st
st.set_page_config(
    page_title="Prescription Parser",
    layout="wide",
    initial_sidebar_state="collapsed",
)
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Default CSS if file not found
        st.markdown("""
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
        }
        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .custom-table th {
            background-color: #005f6b;
            color: white;
        }
        .custom-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        </style>
        """, unsafe_allow_html=True)

try:
    local_css("styles.css")
except:
    pass

# Model definitions
class MedicationItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str

class PrescriptionInformations(BaseModel):
    """Information about a prescription."""
    patient_name: str = Field(description="Patient's name")
    patient_age: int = Field(description="Patient's age")
    patient_gender: str = Field(description="Patient's gender")
    doctor_name: str = Field(description="Doctor's name")
    doctor_license: str = Field(description="Doctor's license number")
    prescription_date: datetime = Field(description="Date of the prescription")
    medications: List[MedicationItem] = []
    additional_notes: str = Field(description="Additional notes or instructions")

# Function to load and encode images
def load_images(inputs: dict) -> dict:
    """Load images from files and encode them as base64."""
    image_paths = inputs["image_paths"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

# Function to process images with vision model
@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with images and prompt."""
    model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")
    image_urls = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in inputs['images']]
    prompt = """
    You are an expert medical transcriptionist specializing in deciphering and accurately transcribing handwritten medical prescriptions. Your role is to meticulously analyze the provided prescription images and extract all relevant information with the highest degree of precision.

    Here are some examples of the expected output format:

    Example:
    Patient's full name: Jane Roe
    Patient's age: 60/60y
    Patient's gender: F/Female
    Doctor's full name: Dr. John Doe
    Doctor's license number: XYZ654321
    Prescription date: 2023-05-10
    Medications:
    - Medication name: Metformin
      Dosage: 850 mg
      Frequency: Once a day
      Duration: 30 days
    Additional notes: 
    - Monitor blood sugar levels daily.
    - Avoid sugary foods.

    Your job is to extract and accurately transcribe the following details from the provided prescription images:
    1. Patient's full name
    2. Patient's age (handle different formats like "42y", "42yrs", "42", "42 years")
    3. Patient's gender
    4. Doctor's full name
    5. Doctor's license number
    6. Prescription date (in YYYY-MM-DD format)
    7. List of medications including:
       - Medication name
       - Dosage
       - Frequency
       - Duration
    8. Additional notes or instructions. Provide detailed and enhanced notes using bullet points. Organize the notes in clear bullet points for better readability.
        - Provide detailed and enhanced notes using bullet points.
        - If there are headings or categories within the notes, ensure the bullet points are organized under those headings.
        - Use clear and concise language to enhance readability.
        - Ensure the notes are structured in a way that makes them easy to follow and understand.

    Important Instructions:
    - Before extracting information, enhance the image for better readability if needed. Use techniques such as adjusting brightness, contrast, or applying filters to improve clarity.
    - Ensure that each extracted field is accurate and clear. If any information is not legible or missing, indicate it as 'Not available'. 
    - Do not guess or infer any information that is not clearly legible.
    - Do not make assumptions or guesses about missing information. 
    - Pay close attention to details like medication names, dosages, and frequencies. 

    Prescription images:
    {images_content}
    """
    parser = JsonOutputParser(pydantic_object=PrescriptionInformations)
    msg = model.invoke(
        [HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "text", "text": parser.get_format_instructions()},
                *image_urls
            ]
        )],
        temperature=0.5,   
        stop=None,  
    )
    return msg.content

# Main function to process prescription images
def get_prescription_informations(image_paths: List[str]) -> dict:
    parser = JsonOutputParser(pydantic_object=PrescriptionInformations)
    vision_prompt = """
    Given the images, provide all available information including:
    - Patient's name, age, and gender
    - Doctor's name and license number
    - Prescription date
    - List of medications with name, dosage, frequency, and duration
    - Additional notes or instructions
    Note: If portions of the image are not clear then leave the values as empty. Do not make up the values.
    """
    vision_chain = load_images_chain | image_model | parser
    return vision_chain.invoke({'image_paths': image_paths, 'prompt': vision_prompt})

# Utility function to clean up temporary files
def remove_temp_folder(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains

# Main Streamlit app
def main():
    # Check if we're in iframe mode
    in_iframe = st.query_params.get("iframe", "false") == "true"
    
    if in_iframe:
        # Simplified header for iframe mode
        st.header('Prescription Parser')
    else:
        # Full header for standalone mode
        st.title('Shree Guru Clinic')
        st.header('Medical Prescription Parsing')
    
    uploaded_file = st.file_uploader("Upload a Prescription image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = uploaded_file.name.split('.')[0].replace(' ', '_')
        output_folder = os.path.join(".", f"Check_{filename}_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)

        check_path = os.path.join(output_folder, uploaded_file.name)
        with open(check_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.expander("Prescription Image", expanded=False):
            st.image(uploaded_file, caption='Uploaded Prescription Image.', use_column_width=True)

        with st.spinner('Processing Prescription...'):  
            try:
                final_result = get_prescription_informations([check_path])
                
                # Process and display results
                if 'additional_notes' in final_result:
                    additional_notes = final_result['additional_notes']
                    # Format additional notes as bullet points
                    if isinstance(additional_notes, list):
                        formatted_notes = "<br> ".join(additional_notes)
                    else:
                        formatted_notes = additional_notes.replace("\n", "<br> ")
                    final_result['additional_notes'] = f"<ul><li>{formatted_notes}</li></ul>"

                # Convert final_result to a list of tuples for DataFrame creation
                data = [(key, final_result[key]) for key in final_result if key != 'medications']
                df = pd.DataFrame(data, columns=["Field", "Value"])

                # Display the DataFrame with custom styling
                st.markdown("<h3>Patient and Doctor Information</h3>", unsafe_allow_html=True)
                st.write(df.to_html(classes='custom-table', index=False, escape=False), unsafe_allow_html=True)

                # Display medications in a separate table with custom styling
                if 'medications' in final_result and final_result['medications']:
                    st.markdown("<h3>Medications</h3>", unsafe_allow_html=True)
                    medications_df = pd.DataFrame(final_result['medications'])
                    st.write(medications_df.to_html(classes='custom-table', index=False, escape=False), unsafe_allow_html=True)
                
                # Download option
                st.download_button(
                    label="Download as CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f'prescription_{timestamp}.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
            finally:
                # Delete temp folder
                remove_temp_folder(output_folder)

if __name__ == '__main__':
    main()