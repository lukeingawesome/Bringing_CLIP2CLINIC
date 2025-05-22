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
Task: Given a specific finding or disease and a chest X-ray report, remove the sentences relevant to that finding or disease.

Context:
Lung lesion: Refers to nodule or mass.
Pleural other: Refers to pleural thickening.

Example:
Finding: Lung Lesion
Report: No pneumothorax is observed. No pleural effusion is observed. No evidence of hemorrhage is observed in the lung or mediastinum. Emphysema is severe. The heart size is normal. A complex of nodule and large bullae is present in the axillary region of the right upper lobe. 
Expected Output: No pneumothorax is observed. No pleural effusion is observed. No evidence of hemorrhage is observed in the lung or mediastinum. Emphysema is severe. The heart size is normal.
"""

# Define the response function using Vertex AI
def get_vertexai_response(finding, report):
    prompt = f"{instruction}\n\nFinding: {finding}\nReport: {report}\nExpected Output:"
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
    rep = af.loc[i, 'result2']
    finding = af.loc[i, 'chosen']
    try:
        af.loc[i, 'result3'] = get_vertexai_response(finding=finding, report=rep)
        af.loc[i, 'error'] = 0
    except Exception as e:
        af.loc[i, 'error'] = 1
        logger.error(f"Error processing row {i}: {e}")
        continue

    if i % 1000 == 0:
        af.to_csv('<Your Output CSV Path>', index=False)

# Save final output
af.to_csv('<Your Output CSV Path>', index=False)
