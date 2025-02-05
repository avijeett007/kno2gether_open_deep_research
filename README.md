# Deep Research Implementation with Smol Agents

> This is a fork of [Huggingface's Smol Agents Deep Research Implementation](https://github.com/huggingface/smolagents/tree/8b02821ac2c73003ea0f533cee0f8e3852d881d9/examples/open_deep_research)

## Watch DeepDive Video

Watch the DeepDive Tutorial on YouTube:

<p align="center">
    <a href="https://youtu.be/V8I6mseJcEc">
        <img src="https://img.youtube.com/vi/V8I6mseJcEc/0.jpg" alt="Deep Research Implementation Tutorial" width="560" height="315">
    </a>
</p>

<p align="center">
    <a href="https://www.youtube.com/channel/UCxgkN3luQgLQOd_L7tbOdhQ?sub_confirmation=1">
        <img src="https://img.shields.io/badge/Subscribe-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Subscribe">
    </a>
</p>

## Introduction

Deep Research Implementation is an AI-powered research assistant built using Huggingface's Smol Agents framework. It enables automated deep research capabilities by combining web browsing, document analysis, and intelligent question answering.

## Key Features

- **Intelligent Web Search**: Advanced web browsing with context awareness
- **Document Analysis**: Support for PDF, Excel, and other document formats
- **Multi-Model Support**: Compatible with GPT-4, Llama, and other LLMs
- **Concurrent Processing**: Multi-threaded question processing
- **Visual Analysis**: Support for image and visual content analysis
- **Automated Research**: Comprehensive research automation tools

## Technical Features

- **Web Browser Integration**: Using SimpleTextBrowser with configurable viewport
- **Document Processing**: TextInspectorTool for various file formats
- **Agent Hierarchy**: Managed agents with specialized capabilities
- **API Integration**: SerpAPI for web searches
- **Concurrent Execution**: ThreadPoolExecutor for parallel processing

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/deep-research-implementation.git
cd deep-research-implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a .env file with:
```env
HF_TOKEN=your_huggingface_token
SERPAPI_API_KEY=your_serpapi_key
OPENAI_API_KEY=your_openai_key
```

4. Test a single question:
```bash
python test_question.py --question "What is quantum computing?" --model-id gpt-4o
```

## System Requirements

- Python 3.10+
- 16GB RAM (32GB recommended for open source models)
- Storage: 10GB minimum
- Required API Keys:
  - Hugging Face Token
  - SerpAPI Key
  - OpenAI API Key

## Available Models

- `gpt-4o`: GPT-4 Turbo (Preview)
- `gpt-4`: Standard GPT-4
- `o3-mini`: GPT-4 Vision
- `llama-3`: Open source Llama model
- `qwen-coder-32B`: Open source Qwen model

## Project Structure

```
deep-research-implementation/
├── run.py                    # Main execution script
├── test_question.py          # Single question testing
├── requirements.txt          # Python dependencies
├── Guide.md                  # Detailed setup guide
└── scripts/                  # Helper scripts
    ├── text_inspector_tool.py
    ├── visual_qa.py
    ├── text_web_browser.py
    └── reformulator.py
```

## Features in Detail

1. **Web Research**
   - Advanced web browsing capabilities
   - Archive search support
   - Configurable viewport size
   - Custom user agent handling

2. **Document Analysis**
   - PDF processing
   - Excel file analysis
   - Text extraction
   - Visual content processing

3. **Agent Management**
   - Hierarchical agent structure
   - Tool-calling capabilities
   - Managed agent prompts
   - Planning intervals

4. **Concurrent Processing**
   - Multi-threaded execution
   - Progress tracking
   - Result aggregation
   - Error handling

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### Community and Support
- Join our community: [Huggingface Forum](https://discuss.huggingface.co/)
- Documentation: [Smol Agents Documentation](https://huggingface.co/docs/smolagents)

### Hosting Partners
- [Kamatera - Get $100 Free VPS Credit](https://knolabs.biz/100-dollar-free-credit)
- [Hostinger - Additional 20% Discount](https://knolabs.biz/20-Percent-Off-VPS)

## Conclusion

Deep Research Implementation showcases the power of Smol Agents in automating complex research tasks. By combining various AI capabilities with web browsing and document analysis, it provides a robust platform for automated research and question answering.

Happy researching! 
