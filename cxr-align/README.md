# CXR-Align Benchmark Dataset Generation Pipeline

This repository contains the pipeline for generating the CXR-Align Benchmark dataset, which processes chest X-ray (CXR) reports through multiple refinement steps to create high-quality, standardized reports for benchmarking.

## Prerequisites

- Python 3.x
- Google Cloud Platform account with Vertex AI API enabled
- Required Python packages:
  - google-cloud-aiplatform
  - pandas
  - tqdm
  - vertexai

## Setup

1. Set up Google Cloud credentials:
   - Create a service account and download the credentials JSON file
   - Set the environment variable: `GOOGLE_APPLICATION_CREDENTIALS="<path_to_your_credentials.json>"`
   - Update the following in each script:
     - `PROJECT_ID`: Your Google Cloud project ID
     - `LOCATION`: Your Vertex AI region
     - Input/output CSV paths

## Pipeline Steps

### Step 1: Initial Report Processing
`step1.py` processes the raw CXR reports to:
- Split reports into individual sentences
- Separate findings and view positions
- Remove non-finding information
- Standardize the format of findings

Input: CSV file with 'report' column containing raw CXR reports
Output: CSV file with processed reports in the 'result' column

### Step 2: Temporal Reference Removal
`step2.py` refines the reports by:
- Removing references to prior reports
- Eliminating comparison terms
- Focusing on current findings only
- Preserving medical terminology and negative statements

Input: CSV file with 'result' column from Step 1
Output: CSV file with refined reports in the 'result2' column

### Step 3: Finding-Specific Processing
`step3.py` performs targeted processing:
- Removes sentences related to specific findings or diseases
- Handles special cases like lung lesions and pleural conditions
- Creates the final "omitted" version of reports

Input: CSV file with 'result2' column from Step 2 and 'chosen' column with target findings
Output: CSV file with final processed reports in the 'result3' column

## Usage

1. Prepare your input CSV file with the raw CXR reports in the 'report' column
2. Run the pipeline in sequence:
   ```bash
   python step1.py
   ```
3. Review and remove any errors from the output
4. Continue with the remaining steps:
   ```bash
   python step2.py
   python step3.py
   ```
5. Use the final output as your "omitted" report dataset

## Error Handling

- Each step includes error logging
- Failed entries are marked with error=1 in the output CSV
- Regular checkpoints are saved every 1000 entries
- Review and clean error entries before proceeding to the next step

## Notes

- The pipeline uses Google's Gemini 1.5 Flash model via Vertex AI
- Processing is done in batches with progress tracking
- Each step builds upon the previous step's output
- The final output maintains the original medical context while removing specified findings
