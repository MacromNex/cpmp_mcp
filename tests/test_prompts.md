# MCP Integration Test Prompts

## Tool Discovery Tests

### Prompt 1: List All Tools
"What MCP tools are available for cyclic peptides? Give me a brief description of each."

**Expected**: List of 15 tools with descriptions including:
- Job management tools (get_job_status, get_job_result, etc.)
- Sync tools (predict_single_assay_permeability, etc.)
- Submit tools (submit_preprocess_data, etc.)
- Validation tools (validate_cyclic_peptide_smiles)

### Prompt 2: Tool Details
"Explain how to use the predict_single_assay_permeability tool, including all parameters."

**Expected**: Detailed explanation of the tool including input parameters (smiles, assay_type) and expected output format.

## Sync Tool Tests

### Prompt 3: Property Calculation - Single Assay
"Calculate PAMPA permeability for this cyclic peptide SMILES: cyclo(GRGDSP)"

**Expected**: Permeability prediction results within 30 seconds

### Prompt 4: Property Calculation - All Assays
"Calculate permeability for all available assays for the cyclic peptide SMILES: cyclo(RGDFV)"

**Expected**: Results for multiple assays (PAMPA, Caco-2, etc.)

### Prompt 5: Data Preprocessing
"Preprocess this cyclic peptide data: SMILES=cyclo(GRGDSP), molecular_weight=677.7"

**Expected**: Preprocessed data with additional calculated features

### Prompt 6: SMILES Validation
"Validate if 'cyclo(Gly-Arg-Gly-Asp-Ser-Pro)' is a valid cyclic peptide representation"

**Expected**: Validation result with true/false and explanation

### Prompt 7: Error Handling - Invalid SMILES
"Calculate permeability for an invalid SMILES 'not_a_valid_smiles_string'"

**Expected**: Clear error message explaining invalid SMILES

### Prompt 8: Server Info
"What information can you provide about the cycpep-tools server?"

**Expected**: Server version, available tools count, description

## Submit API Tests (Long-Running Tasks)

### Prompt 9: Submit Data Preprocessing
"Submit a data preprocessing job for the cyclic peptide SMILES: cyclo(Ala-Gly-Pro-Phe)"

**Expected**: Job submission response with job_id

### Prompt 10: Submit Single Assay Prediction
"Submit a PAMPA permeability prediction job for cyclo(GRGDSP)"

**Expected**: Job submission response with job_id

### Prompt 11: Submit All Assays Prediction
"Submit predictions for all assays for the cyclic peptide cyclo(RGDFV)"

**Expected**: Job submission response with job_id

### Prompt 12: Submit Batch Analysis
"Submit a batch analysis job for these cyclic peptides: cyclo(GRGDSP), cyclo(RGDFV), cyclo(YIGSR)"

**Expected**: Batch job submission with job_id

### Prompt 13: Check Job Status
"Check the status of job <job_id_from_previous_test>"

**Expected**: Job status information (pending/running/completed/failed)

### Prompt 14: Get Job Results
"Get the results for job <completed_job_id>"

**Expected**: Actual computation results or error message

### Prompt 15: View Job Logs
"Show me the last 20 lines of logs for job <job_id>"

**Expected**: Log output from the job execution

### Prompt 16: List All Jobs
"List all submitted jobs and their current status"

**Expected**: List of jobs with IDs, status, and timestamps

### Prompt 17: Cancel Job (if applicable)
"Cancel the running job <running_job_id>"

**Expected**: Confirmation of job cancellation

## End-to-End Scenarios

### Prompt 18: Full Workflow
"For the cyclic peptide sequence GRGDSP:
1. Validate it as a cyclic peptide
2. Convert to appropriate representation
3. Calculate PAMPA permeability
4. Submit a full analysis job
Summarize all results."

**Expected**: Complete workflow execution with results from each step

### Prompt 19: Drug-likeness Assessment
"Assess the drug-likeness of cyclo(Pro-Leu-Gly-Phe-Ala):
- Preprocess the data
- Predict permeability for all assays
- Summarize if it meets cyclic peptide drug criteria"

**Expected**: Comprehensive drug-likeness analysis

### Prompt 20: Virtual Screening Simulation
"I want to screen these cyclic peptides for permeability:
- cyclo(GRGDSP)
- cyclo(RGDFV)
- cyclo(YIGSR)
Calculate PAMPA permeability for each and rank them."

**Expected**: Permeability results for all peptides with ranking

### Prompt 21: Batch Processing Workflow
"Submit a batch analysis for multiple cyclic peptides and then check the status until completion"

**Expected**: Full batch workflow from submission to results retrieval

## Error Handling and Edge Cases

### Prompt 22: Invalid Job ID
"Check the status of job 'nonexistent_job_id'"

**Expected**: Clear error message about invalid job ID

### Prompt 23: Empty Input
"Calculate permeability for an empty SMILES string ''"

**Expected**: Appropriate error handling for empty input

### Prompt 24: Malformed Input
"Submit a job with malformed input data"

**Expected**: Validation error with helpful message

## Performance Tests

### Prompt 25: Multiple Concurrent Requests
"Calculate PAMPA permeability for 5 different cyclic peptides simultaneously"

**Expected**: All requests handled correctly without interference

### Prompt 26: Large Batch Job
"Submit a batch analysis for 10+ cyclic peptides"

**Expected**: Successful handling of larger batch sizes

## Test Results Documentation

For each test prompt:
- Record the actual response time
- Note any errors or unexpected behaviors
- Verify the response format matches expectations
- Document any issues found for debugging