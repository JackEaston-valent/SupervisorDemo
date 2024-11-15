## Environment Setup

### Python environment

I like to use pyenv and create a virtual environment with:

`> pyenv virtualenv <installed python version> <environment name>`

Then you can set the environment of your local folder with:

`> pyenv local <environment name>`

Then you need to install the required dependencies:

`> pip install langchain langchain-openai langchain_community langgraph langchain-chroma langchain_experimental matplotlib`

### API Keys

The `dotenv` python library reads in keys from a file called `.env` that you need to create. This is where you would copy in your API keys.

Example `.env` file:
```
OPENAI_API_KEY=sk-proj-12345
TAVILY_API_KEY=tvly-12345
LANGCHAIN_API_KEY=lsv2_1234
```

## Example call
`> python supervisor.py --message "What are my options for authentication software?"`