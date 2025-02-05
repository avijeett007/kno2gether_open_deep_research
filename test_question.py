import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from smolagents import (
    LiteLLMModel,
    CodeAgent,
    ToolCallingAgent,
    ManagedAgent,
    MANAGED_AGENT_PROMPT,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.visual_qa import visualizer
from scripts.text_web_browser import (
    SimpleTextBrowser,
    SearchInformationTool,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
)
from scripts.reformulator import prepare_response
from scripts.run_agents import get_document_description
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="Chat templates should be in a 'chat_template.jinja' file")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # This will suppress the ffmpeg warning

# Load environment variables
load_dotenv(override=True)

# Add AUTHORIZED_IMPORTS from run.py
AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]

# Create downloads folder
os.makedirs("downloads_folder", exist_ok=True)

def create_agent_hierarchy(model):
    # Setup browser config
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
    
    # Create browser and tools
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)
    
    WEB_TOOLS = [
        SearchInformationTool(browser),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    # First create the base ToolCallingAgent with only the supported parameters
    base_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
    )

    # Then wrap it in a ManagedAgent to add the name, description and other features
    text_webbrowser_agent = ManagedAgent(
        agent=base_agent,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        managed_agent_prompt=MANAGED_AGENT_PROMPT
        + """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    )

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )
    
    return manager_agent

def select_model(model_id: str):
    """
    Helper function to select and configure the appropriate model.
    
    Args:
        model_id: String identifier for the model. Options:
            - "o1": GPT-4 Turbo (Preview)
            - "gpt-4o": Standard GPT-4
            - "o3-mini": GPT-4 Vision
            - "llama-3": Open source Llama model
            - "qwen-coder-32B": Open source Qwen model
    """
    return LiteLLMModel(model_id)

def get_single_file_description(file_path, question, visual_inspection_tool, document_inspection_tool):
    try:
        if not os.path.exists(file_path):
            return f"\nWARNING: File {file_path} does not exist. Proceeding without file content."
            
        description = get_document_description(file_path, question, document_inspection_tool)
        return f"""
Here is the file that you need to use: {file_path}
Here is a description of its content:
{description}
"""
    except Exception as e:
        return f"\nWARNING: Could not process file {file_path}. Error: {str(e)}. Proceeding without file content."

def answer_single_question(question, model_id="o1", attached_file=None):
    # Initialize model
    model = select_model(model_id)
    
    # Create agent
    agent = create_agent_hierarchy(model)
    
    # Create document inspection tool
    document_inspection_tool = TextInspectorTool(model, 100000)
    
    # Prepare the question
    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer.
Here is the task:
""" + question

    # Add file information if provided
    if attached_file:
        prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
        prompt_use_files += get_single_file_description(
            attached_file, question, visualizer, document_inspection_tool
        )
        augmented_question += prompt_use_files

    # Record start time
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Run agent
        final_result = agent.run(augmented_question)
        agent_memory = agent.write_memory_to_messages(summary_mode=True)
        final_answer = prepare_response(augmented_question, agent_memory, reformulation_model=model)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
        
    # Record end time
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "question": question,
        "answer": final_answer,
        "start_time": start_time,
        "end_time": end_time
    }

if __name__ == "__main__":
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Test Deep Research with a specific question')
    parser.add_argument('--question', type=str, required=True, help='The question to answer (use quotes for multiple words)')
    parser.add_argument('--model-id', type=str, default='o1', 
                      choices=['o1', 'gpt-4o', 'o3-mini', 'llama-3', 'qwen-coder-32B'],
                      help='Model to use for answering')
    parser.add_argument('--file', type=str, help='Path to an attached file (optional)')
    
    try:
        args = parser.parse_args()
    except Exception:
        # If normal parsing fails, try to reconstruct the command with proper quoting
        import sys
        import shlex
        
        # Join all args after the script name and properly quote them
        command = ' '.join(sys.argv[1:])
        try:
            # Parse the command respecting quotes
            parsed_args = shlex.split(command)
            args = parser.parse_args(parsed_args)
        except Exception as e:
            print("\nError: Please make sure to put your question in quotes if it contains spaces.")
            print('Example: python test_question.py --question "What is the capital of France?" --model-id o1')
            sys.exit(1)
    
    print(f"\nUsing model: {args.model_id}")
    print(f"Processing question: {args.question}")
    if args.file:
        print(f"With attached file: {args.file}")
        
    result = answer_single_question(
        question=args.question,
        model_id=args.model_id,
        attached_file=args.file
    )
    
    if result:
        print("\nQuestion:", result["question"])
        print("\nAnswer:", result["answer"])
        print("\nTime taken:", result["start_time"], "to", result["end_time"])
    else:
        print("\nError: Failed to get an answer.")