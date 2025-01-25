# PDF Question Answering System

This use-case is about a document QA system using LangChain and OpenAI's GPT-3.5-turbo model.


## Installation
```bash
pip install langchain-openai PyPDF2
```

## Usage
```python
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

OPENAI_API_KEY = "your-key-here"
qa_chain = setup_qa_chain(OPENAI_API_KEY)

# trigger questions from below code snippet
response = qa_chain.invoke({
    "context": your_text,
    "question": your_question
})
```

## Features
- PDF text extraction  
- Context-aware question answering
- Configurable model parameters