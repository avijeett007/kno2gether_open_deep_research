# Deep Research Implementation Setup and Testing Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Testing Original Implementation](#testing-original-implementation)
- [Testing Single Questions](#testing-single-questions)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Usage](#advanced-usage)

## Prerequisites

### Required API Keys
1. **Hugging Face Token (HF_TOKEN)**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read access
   - This is used for accessing the GAIA dataset and models

2. **SerpAPI Key (SERPAPI_API_KEY)**
   - Sign up at https://serpapi.com/
   - Get your API key from the dashboard
   - This is used for web searches

3. **OpenAI API Key (OPENAI_API_KEY)**
   - Sign up at https://platform.openai.com/
   - Create a new API key
   - This is used for the main language model capabilities

### System Requirements
- Python 3.10 or higher (recommended)
- Git
- 16GB RAM minimum (recommended 32GB for open source models)
- Storage space: at least 10GB free

## Environment Setup

1. **Clone the Repository**
```bash
git clone [repository-url]
```

2. **Create Conda Environment**
```bash
# Create new conda environment
conda create -n deepsearch python=3.10
conda activate deepsearch

# Install requirements
pip install -r requirements.txt
```

3. **Setup Environment Variables**
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_token
SERPAPI_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

4. **Create Required Directories**
```bash
mkdir downloads_folder
mkdir output
mkdir output/validation
```

## Testing Original Implementation

### 1. Basic Test Run
```bash
python run.py --model-id gpt-4o --run-name first_test --concurrency 1
```
This will:
- Use the gpt-4o model (GPT-4 Turbo Preview)
- Save results in `output/validation/first_test.jsonl`
- Process one question at a time

### 2. Testing with Different Models
```bash
# Test with GPT-4
python run.py --model-id gpt-4o --run-name gpt4_test --concurrency 1

# Test with open source model
python run.py --model-id llama-3 --run-name llama_test --concurrency 1
```

## Testing Single Questions

### 1. Basic Question Testing
Important: Always put questions with spaces in quotes!

```bash
# Correct usage:
python test_question.py --question "What is the capital of France?" --model-id gpt-4o

# Wrong usage (will cause error):
python test_question.py --question What is the capital of France? --model-id gpt-4o
```

### 2. Available Models
- `gpt-4o`: GPT-4 Turbo (Preview)
- `gpt-4o`: Standard GPT-4
- `o3-mini`: GPT-4 Vision
- `llama-3`: Open source Llama model
- `qwen-coder-32B`: Open source Qwen model

### 3. Testing with File Attachments
```bash
# With a PDF file
python test_question.py \
    --question "Summarize this document" \
    --model-id gpt-4o \
    --file "path/to/document.pdf"

# With an Excel file
python test_question.py \
    --question "What's the total revenue?" \
    --model-id gpt-4o \
    --file "path/to/spreadsheet.xlsx"
```

## Common Issues and Solutions

### 1. Question Parsing Errors
If you see "unrecognized arguments" error:
- Make sure to put multi-word questions in quotes
- Check for any special characters in the question
- Use proper escaping if needed

### 2. File Not Found Errors
- Check if the file path is correct
- Make sure you have necessary permissions
- For GAIA dataset files, ensure you have HF_TOKEN set

### 3. API Errors
- Verify your API keys in .env
- Check your internet connection
- Ensure you're not hitting rate limits

### 4. Memory Issues
If you encounter memory errors:
```bash
# Reduce batch size
python run.py --model-id gpt-4o --run-name test --concurrency 1
```

## Advanced Usage

### 1. Custom Browser Configuration
The default browser config is set to:
```python
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}
```

### 2. Debugging Tips
1. Enable verbose output:
```python
verbosity_level=2  # In agent configuration
```

2. Monitor JSONL output:
```bash
tail -f output/validation/your_run_name.jsonl
```

Remember to check the output directory for results after each run.