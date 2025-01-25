# LangChain Text Summarizer

This usecase is a simple text summarization tool built using LangChain composable chain architecture and OpenAI's GPT-3.5 model.


## Setup
1. Install necessary packages
```bash
pip install langchain-openai langchain-core
```
2. Set your OpenAI API key in the code:
```python
OPENAI_API_KEY = "your-api-key-here"
```

## Structure
- `create_summaries()`: Creates and returns a LangChain processing chain
- Main execution block: Contains example text and demonstrates usage

## Features
- Uses PromptTemplate for consistent formatting
- Implements RunnablePassthrough for chain composition
- Leverages ChatOpenAI for text generation
- Returns structured key points from input text