# News Summarization and Sentiment Analysis

## 1. Project Setup
Steps to install and run the application.

### Prerequisites
- Python 3.8 or higher.  
- Pip (Python package manager).  

### Installation
*1. Clone the repository:*
   ```bash
   git clone <repository-url>
   cd News_Summarization
```
*2. Install dependencies:*
   ```bash
   pip install -r requirements.txt
```
   

*3. Set up API keys:*
- Obtain API keys for:
  - Hugging Face (for Inference API) → Due to less computational resources
  - OpenRouter (for qwq-32B)
  - Google Gemini (Gemma-27B)
  - Groq API key (llama3.3-70B-versatile)
- Add the keys to the `.env` file

*4. Running the Application:*
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

## Model Details
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

