from google.cloud import aiplatform
import pandas as pd
from tqdm import tqdm
import logging
import os
from vertexai.preview.generative_models import GenerativeModel
from vertexai import init
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<Your Credentials Path>"
# Configure Vertex AI
PROJECT_ID = "<Your Project ID>"         # Replace with your Google Cloud project ID
LOCATION = "<Your Region>"               # Change this to your Vertex AI location (region)
MODEL_NAME = "gemini-1.5-flash-001"    # Gemini model available via Vertex AI

init(project=PROJECT_ID, location=LOCATION)

# Initialize Gemini model via Vertex AI
model = GenerativeModel("gemini-1.5-flash-001")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instructions to the model
instruction = """
You are an expert chest X-ray (CXR) radiologist familiar with radiologic reports. Your task is to rewrite the given radiology reports by removing all references to prior reports or comparisons, while preserving the original structure as much as possible.
Input: A radiology report for a chest X-ray (CXR).
Output: A revised CXR report focusing solely on current medical findings, excluding references to prior reports, comparisons, and irrelevant details.
Guidelines:
Remove Comparisons: Eliminate any terms or phrases that suggest a comparison, such as "compared to," "in comparison with," "change", "cleared", "constant", "decrease", "elevate", "expand", "improve", "decrease", "increase", "persistent", "reduce", "remove", "resolve", "stable", "worse", "new", etc.
Focus on Current Findings: Ensure the report only describes the current state of the patient's lungs and related structures.
Preserve Medical Context: Maintain the original medical terminology and descriptions of abnormalities.
Retain Negations: Keep any negative statements about the absence of abnormalities.

Example 1:
Original: The left apex has not been included on this radiograph. The ET tube terminates 3.9 cm above the carina. The NG tube terminates in the stomach. Surgical clips and a faint metallic coil project over the chest. A left PICC terminates in the mid SVC. EKG leads overlie the chest wall. The lung volumes are low. There are persistent bilateral mid and lower zone hazy opacities. There are persistent bilateral hilar and perihilar linear opacities. No significant interval change is observed in the lung opacities. Bilateral pleural effusions are present. The right pleural effusion is greater than the left. No pneumothorax is observed on the right. No cardiomegaly is present. No interval change is observed in the mediastinal silhouette. No significant interval change is observed in the bony thorax.  
Revised: The left apex has not been included on this radiograph. The ET tube terminates 3.9 cm above the carina. The NG tube terminates in the stomach. Surgical clips and a faint metallic coil project over the chest. A left PICC terminates in the mid SVC. EKG leads overlie the chest wall. The lung volumes are low. There are persistent bilateral mid and lower zone hazy opacities. There are bilateral hilar and perihilar linear opacities. Bilateral pleural effusions are present. The right pleural effusion is greater than the left. No pneumothorax is observed on the right. No cardiomegaly is present.

Example 2:
Original: Compared to chest radiographs ___ through ___. Improved nodular opacity in right upper lobe. Resolved pneumothorax. Pleural effusion in left lung is slightly increased. A right PIC line is in the low SVC. The lungs are clear. The heart size is top-normal. No pleural abnormality is observed. An indwelling feeding tube passes into the stomach and out of view. An ET tube is in standard placement.  
Revised: Nodular opacity in right upper lobe. No pneumothorax. Pleural effusion in left lung.  A right PIC line is in the low SVC. The heart size is top-normal. No pleural abnormality is observed. An indwelling feeding tube passes into the stomach and out of view. An ET tube is in standard placement.  
"""

# Define the response function using Vertex AI
def get_vertexai_response(report):
    prompt = f"{instruction}\n\nOriginal: {report}\nRevised:"
    response = model.generate_content(prompt, generation_config={
        "temperature": 0.0,
        "max_output_tokens": 320
    })
    return response.text

# Load dataset
df = pd.read_csv('<Your CSV Path>')
af = df.copy()

# Iterate over dataset and apply the Gemini Vertex AI model
for i in tqdm(range(len(df))):
    rep = af['result'][i].replace('\n', ' ')
    try:
        af.loc[i, 'result2'] = get_vertexai_response(report=rep)
    except Exception as e:
        af.loc[i, 'error'] = 1
        logger.error(f"Error processing row {i}: {e}")
        continue
    if i % 1000 == 0:
        af.to_csv('<Your Output CSV Path>', index=False)

# Save final output
af.to_csv('<Your Output CSV Path>', index=False)
