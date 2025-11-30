# Macro Financial Forecasting

## Description

Agentic pipeline for extracting factors and forecasting next-day returns from macroeconomic and financial news (Bloomberg). The project combines streaming RSS ingestion, Pydantic data models, FinBERT/DeBERTa for classification and sentiment, and LangGraph + LLMs for summarization and prediction orchestration.


https://github.com/user-attachments/assets/898d0a38-6c6f-48e5-ac0e-ba22cf5c8c83


## Deployment Information

-   **Project Slug:** `agentic-macro-fincast`
-   **Deployment URL:** `https://[cloudfront-domain]/agentic-macro-fincast`
-   **Main File:** `streamlit.py`

## Environment Variables Required

-   `OPENAI_API_KEY`: OpenAI API key (used by LangGraph / OpenAI-backed LLMs)
-   `LLM_MODEL`: Default LLM model id (e.g. `gpt-5-nano-2025-08-07`)
-   `LANGGRAPH_MODEL`: Model id used by LangGraph (e.g. `gpt-5-nano`)
-   `INDUSTRIES`: Comma-separated list of allowed industry labels
-   `DATASET_NAME`: Hugging Face dataset name (default: `danidanou/BloombergFinancialNews`)
-   `DATASET_DIR`: Local dataset/cache directory (default: `../data`)
-   `RSS_FEEDS`: Comma-separated RSS feed URLs
-   `FINBERT_MODEL`: FinBERT model id (default: `ProsusAI/finbert`)
-   `DEBERTA_MODEL`: DeBERTa model id (default: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`)

## Local Setup

### Using Conda (Recommended)

```bash
# Create the Conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate agentic-macro-fincast

# Run the Streamlit app
streamlit run streamlit.py
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit.py
```

## Notes

- The project also provides an `environment.yml` (Conda) for reproducible environments. Prefer creating the Conda environment for binary dependencies (PyTorch, pyarrow, pandas) and then installing pip packages.
- If you plan to use the Hugging Face Hub or the OpenAI SDK directly, add `huggingface-hub` and `openai` to your environment.

## Docker
Update config.env with your OPENAI_API_KEY before running container.
```bash
docker build --platform linux/amd64 -t agentic-macro-fincast:test .
docker run -p 8501:8501 --env-file config.env agentic-macro-fincast:test
#open http://localhost:8501/agentic-macro-fincast
```

## Contributors

- Chuan Bin Phoe
- Neaton Ang

## Langgraph pipeline
Framework can be extended:
- Multi-agent
- New data sources, prediction models
- Using Agent for path planning (decision on next step)

![WhatsApp Image 2025-11-28 at 10 37 56_4def2b09](https://github.com/user-attachments/assets/e14b7904-1976-457a-86f9-027b71d59d6d)

