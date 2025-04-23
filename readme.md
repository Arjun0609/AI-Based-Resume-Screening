## Setup

```bash
# Clone the repository
git clone https://github.com/Arjun0609/AI-Based-Resume-Screening.git
cd AI-Based-Resume-Screening

# Create Virtual Environment
python -m venv .venv
.venv/Scripts/activate

# Setup Dataset
# Unzip archive.zip
# Rename archive to dataset

# Install Python dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

# Download all NLTK data
# In Terminal
>>> python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('punkt_tab')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> exit()

# Run the main script
python app.py analyze test_resume.pdf
```
