# 🧬 Multi-Omics Explorer

Streamlit app + FastAPI service for exploring PubMed abstracts and GEO datasets.

---

## 🚀 Features

- PubMed topic modeling using BERTopic
- GEO dataset metadata extraction & visualization
- FastAPI backend to serve PubMed data via REST API

---

## 🔧 Requirements

- Python 3.10+
- Docker

---

## 📦 Installation (Local)

```bash
# Clone the repo
git clone https://github.com/yourusername/multi-omics-explorer.git
cd multi-omics-explorer

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run Streamlit app
streamlit run multi_omics_explorer.py
