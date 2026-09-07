"""
OpenAI_CD_annotations.py — Cognitive Distortion Annotation via Azure OpenAI

Runs a chat-completion prompt (rank-label or multi-label) over patient questions
to identify cognitive distortions, repeating each question `iterations` times to
capture sampling variability. Results are saved as CSV under gpt_annotations/.

Requirements:
    - Environment variable OPENAI_API_KEY set to your Azure OpenAI key
    - Original_data.csv in the working directory, with Id_Number and Patient
      Question as the first two columns

Usage (single run):
    python OpenAI_CD_annotations.py --model_type gpt-4 --prompt_type RLP --temperature 0.5

Arguments:
    --model_type    gpt-4 or gpt-4o
    --prompt_type   RLP (rank-label prompt) or MLP (multi-label prompt)
    --temperature   sampling temperature, e.g. 0.5 or 0.7

Batch runs (all 8 model x prompt x temperature combinations) are submitted via
SLURM using the accompanying job array script:
    sbatch annotate_job.sh

Output:
    OpenAI_CD_annotations/results_<model_type>_<prompt_type>_temp<temperature>.csv
"""


import os
import pandas as pd
import argparse
from openai import AzureOpenAI

# Argument parsing
parser = argparse.ArgumentParser(description="Run Cognitive Distortion Annotation")
parser.add_argument('--model_type', type=str, required=True, help='Model type: gpt-4 or gpt-4o')
parser.add_argument('--prompt_type', type=str, required=True, help='Prompt type: RLP or MLP') #rank label prompt and multilabel prompt
parser.add_argument('--temperature', type=float, required=True, help='Temperature setting: 0.5 or 0.7')
args = parser.parse_args()

# Load the annotated data
input_file = "Original_data.csv"
data = pd.read_csv(input_file, usecols=[0, 1])
data.columns = ["Id_Number", "Patient Question"]

# Set up Azure OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
client = AzureOpenAI(
    api_version="api_version", # write api version
    azure_endpoint="endpoint", # your endpoint link
    api_key=api_key, # your api key
)

# Define the prompts
prompt_RLP = """
As a specialized annotator in psychology with expertise in cognitive distortions, analyze the ```Patient Question``` to identify any 
underlying cognitive distortion(s) from the specified list called **Cognitive Distortions List**. 
Your task is to determine the most dominant cognitive distortion present. If there is a secondary distortion, note it as well. 
In cases where multiple distortions are present, select the most dominant one as the primary label, and if necessary, include one secondary label. 
The response must include at most two labels. If no distortions are found, label the question as **No Distortion**.

---
**Cognitive Distortions List**: [Emotional Reasoning, Overgeneralization, Mental Filter, Should Statements, All or Nothing Thinking, Mind Reading,
                                Fortune Telling, Magnification (Catastrophizing), Personalization, Labeling]
---

**Output Format**: 
- If cognitive distortions are identified, provide them as a comma-separated list, with the most dominant distortion listed first, 
followed by a secondary distortion if applicable. At most two distortions should be listed.
- If no distortions are found, return **No Distortion**.
"""

prompt_MLP = """
As a specialized annotator in psychology with expertise in cognitive distortions, analyze the ```Patient Question``` to identify any underlying 
cognitive distortion(s) from the specified list called **Cognitive Distortions List** and provide results as per the **Output Format** only.

---
**Cognitive Distortions List**: [Emotional Reasoning, Overgeneralization, Mental Filter, Should Statements, All or Nothing Thinking, Mind Reading,
                                Fortune Telling, Magnification (Catastrophizing), Personalization, Labeling]
---

**Output Format**: If any cognitive distortion(s) are identified, list them as a comma-separated list. If no distortion(s) are found, return **No Distortion**.
"""

# Select the appropriate prompt
prompt = prompt_RLP if args.prompt_type == "RLP" else prompt_MLP

# Initialize a list to store the results
results = []

# Number of iterations for each question
iterations = 5

# Process each row in the DataFrame
for _, row in data.iterrows():
    patient_question = row['Patient Question']
    question_id = row['Id_Number'] 

    # Initialize a list to hold results for this question across iterations
    question_results = []
    
    for iteration in range(1, iterations + 1):
        messages = [
            {"role": "system", "content": prompt}, 
            {"role": "user", "content": patient_question},
        ]
        
        try:
            # Make the API call
            response = client.chat.completions.create(
                model=args.model_type,
                messages=messages,
                temperature=args.temperature,
            )
            # Extract raw response
            raw_response = response.choices[0].message.content

            # Save raw response with iteration numbering
            question_results.append({
                f'Iter {iteration}': raw_response
            })
        
        except Exception as e:
            violation_response = 'Violation' # in case of active content filtering on Azureopenai
            
            question_results.append({
                f'Iter {iteration}': violation_response
            })
    
    # Append the results for all iterations of this question to the data collection list
    results.append({
        'Id_Number': question_id,
        'Question': patient_question,
        "Model": args.model_type,
        "Temperature": args.temperature,
        "Prompt_Type": args.prompt_type,
        'Results': question_results
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
output_dir = "gpt_annotations"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"results_{args.model_type}_{args.prompt_type}_temp{args.temperature}.csv")
results_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")

