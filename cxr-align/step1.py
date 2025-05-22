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
Analyze the given radiology report text and split it into sentences, each describing a single radiological finding or view position information. Include both positive and negative findings, as well as view position details, but exclude other non-finding information. Follow these guidelines:

1. Each sentence should contain only one of the following: (exclude the sentence with view position information)
   a) A clear radiological finding
   b) The absence of a specific condition (negative finding)

2. Treat each negative finding (absence of a condition) as a separate observation and split it into its own sentence.

3. Keep sentences that describe view position information, but separate them from findings if they appear in the same original sentence.

4. Exclude sentences that do not describe actual radiological findings or view positions, such as:
   - Procedural details
   - General comments about image quality
   - Patient positioning information (unless it's specifically about the view position)

5. Maintain the meaning and context of the original findings and view positions while splitting.

6. Minor sentence structure changes or addition of necessary words are allowed to ensure clarity.

7. Remove any redundant information and express each finding or view position concisely.

8. Each split sentence should be understandable independently.

9. Avoid using lists or enumerations within a single sentence; instead, create separate sentences for each item.

Example of splitting, including view position, and excluding non-findings:
Original: '1. A single frontal view of the chest is provided. 2. No consolidation, pleural effusion, or pneumothorax is observed in both lungs. 3. The heart size is normal.'
Revised: 'A single frontal view of the chest is provided. No consolidation is observed in both lungs. No pleural effusion is observed in both lungs. No pneumothorax is observed in both lungs. The heart size is normal.'

Process the input text according to these guidelines and return the relevant radiological findings and view position information and do not attach any additional text except the split sentences.
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
    rep = af['report'][i]
    try:
        af.loc[i, 'result'] = get_vertexai_response(report=rep)
    except Exception as e:
        af.loc[i, 'error'] = 1
        logger.error(f"Error processing row {i}: {e}")
        continue
    if i % 1000 == 0:
        af.to_csv('<Your Output CSV Path>', index=False)

# Save final output
af.to_csv('<Your Output CSV Path>', index=False)
