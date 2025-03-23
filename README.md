# News Summarization and Sentiment Analysis

## 🌐 Try it on Hugging Face  
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFCC4D?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/SriArun/NewsSummarization)

## 1. Project Setup
Steps to install and run the application.

### Prerequisites
- Python 3.8 or higher.  
- Pip (Python package manager).

### Directory
![image](https://github.com/user-attachments/assets/c574e620-2f8b-466c-9f50-05d0417faea0)

### Installation
*1. Clone the repository:*
   ```bash
   git clone https://github.com/SriArunM/News_Summarization.git
   cd News_Summarization
```
*2. Create virtual environment:*
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
```
*3. Install dependencies:*
   ```bash
   pip install -r requirements.txt
```
   

*4. Set up API keys:*
- Obtain API keys for:
  - Hugging Face (for Inference API) → Due to less computational resources
  - OpenRouter (for qwq-32B)
  - Google Gemini (Gemma-27B)
  - Groq API key (llama3.3-70B-versatile)
- Add the keys to the `.env` file

*5. Running the Application:*
- Start the FastAPI backend:
  
     ```bash
      python api.py
    ```

The backend will be available at `http://127.0.0.1:8000`.

- Start the Streamlit frontend:

```bash
streamlit run app.py
```
The frontend will be available at `http://localhost:8501`


## 2.Model Details
Explanation of models used for summarization, sentiment analysis, and TTS.

### Summarization Model: facebook/bart-large-cnn (Hugging Face)
- **Purpose**: Summarizes long articles into concise summaries
- **Fallback**: Google GenAI (gemma-3-27b-it) if Hugging Face fails

### Sentiment Analysis
- **Primary Model**: Google GenAI (gemma-3-27b-it)
  - **Purpose**: Analyzes the sentiment of articles (Positive, Negative, Neutral)
- **Fallback**: Hugging Face (finiteautomata/bertweet-base-sentiment-analysis) if Google GenAI fails

### Text-to-Speech (TTS)
- **Primary Model**: gTTS (Google Text-to-Speech)
  - **Purpose**: Converts Hindi text to speech
- **Fallback**: Hugging Face MMS-TTS (facebook/mms-tts-hin) if gTTS fails

### Topic Extraction
- **Primary Model**: Groq (llama-3.3-70b-versatile)
  - **Purpose**: Extracts key topics from articles
- **Fallback**: Google GenAI (gemma-3-27b-it) if Groq fails

