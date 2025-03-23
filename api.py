from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from utils import (
    extract_news,
    generate_comparative_analysis,
    add_topic_overlap_to_coverage_differences,
    generate_company_analysis_in_hindi,
    convert_text_to_speech,
)
import os

app = FastAPI()

class CompanyRequest(BaseModel):
    company_name: str

@app.post("/extract-news/")
def extract_news_api(request: CompanyRequest):
    company_name = request.company_name
    news_articles = extract_news(company_name)

    if not news_articles.get("Articles"):
        raise HTTPException(status_code=404, detail="No articles found")

    # Generate comparative analysis
    updated_news_articles = generate_comparative_analysis(news_articles)

    # Add topic overlap analysis
    updated_news_articles = add_topic_overlap_to_coverage_differences(updated_news_articles)

    # Generate company analysis in Hindi
    sentiment_report = generate_company_analysis_in_hindi(company_name, updated_news_articles)

    audio_file = None
    if sentiment_report:
        audio_file = convert_text_to_speech(sentiment_report, "output.mp3")

    response = {
        "articles": updated_news_articles,
        "sentiment_report": sentiment_report,
        "audio_file": audio_file if audio_file and os.path.exists(audio_file) else None
    }
    return response

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)