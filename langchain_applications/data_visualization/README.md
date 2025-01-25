# AI-Powered Flight Data Visualization

This usecase analyzes flight passenger data using LangChain and OpenAI, generating visualizations with Seaborn.


## Installation
```bash
pip install numpy==1.23.5 matplotlib seaborn langchain-openai
```

## Features
- Loads Seaborn's built-in flights dataset
- Uses OpenAI to suggest optimal visualizations
- Creates three plots:
  - Time series of passenger trends
  - Monthly passenger distribution
  - Average passengers per month
