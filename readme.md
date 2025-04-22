## Setup

```bash
# Clone the repository
git clone https://github.com/Arjun0609/AI-Based-Resume-Screening.git
cd AI-Based-Resume-Screening

# Create Virtual Environment
python -m venv .venv
.venv/Scripts/activate

# Install Python dependencies
pip install -r requirements.txt

# Download all NLTK data
# In Terminal
>>> python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> exit()

# Run the main script
python app.py analyze <resume_path>
# eg:
python app.py analyze test_resume.pdf
```
