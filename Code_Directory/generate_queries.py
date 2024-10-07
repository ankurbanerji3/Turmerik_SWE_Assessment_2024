import os
from openai import OpenAI
import xml.etree.ElementTree as ET
import json

# Set up your OpenAI API key

client = OpenAI(
    # This is the default and can be omitted
    api_key='You API Key',
)

# Function to parse the XML file and extract relevant patient information
def extract_patient_info(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract patient ID (using your XML structure as a guide)
    namespace = {'hl7': 'urn:hl7-org:v3'}
    patient_id = root.find('.//hl7:id', namespace).attrib['extension']

    # Extract patient details to summarize
    patient_info = ET.tostring(root, encoding='unicode')

    return patient_id, patient_info

# Function to summarize patient information using OpenAI API
def summarize_patient_info(patient_info):

    response = client.chat.completions.create(
    model="gpt-4o",  # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a concise medical assistant."},
        {"role": "user", "content": f"Summarize the following patient's medical information in one sentence:\n\n{patient_info}"}
    ],
    max_tokens=150
    )
    summary = response.choices[0].message.content
    # print(type(summary))
    print(summary)
    return summary

# Function to process all XML files in a directory
def process_patient_directory(directory):
    patient_summaries = []

    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_file = os.path.join(directory, filename)
            
            # Extract patient info from XML
            patient_id, patient_info = extract_patient_info(xml_file)

            # Submit the extracted data to OpenAI and get the summary
            summary = summarize_patient_info(patient_info)

            # Add the summary to the dictionary
            patient_summaries.append({
                "_id": patient_id,
                "text": summary
            })

    return patient_summaries

if __name__ == '__main__':
    # Specify the directory where all XML files are stored
    directory_path = 'D:/Patient/synthea_1m_fhir_3_0_May_24/output_1/CCDA'
    
    # Process the directory and get the patient summaries
    patient_summaries = process_patient_directory(directory_path)
    
    # Output the final dictionary with all patient summaries
    print(patient_summaries)
    
    # Optional: Save the patient summaries to a JSON file
    import json
    with open('D:/Patient/queries.json', 'w') as f:
        json.dump(patient_summaries, f, indent=4)