from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from utils import (
    extract_news,
    generate_comparative_analysis,
    add_topic_overlap_to_coverage_differences,
    generate_company_analysis_in_hindi,
    convert_text_to_speech,
)

app = FastAPI()


@app.post("/analyze-news")
async def analyze_news(company_name: str, num_articles: int = 10):
    try:
        # Extract news articles
        news_articles = extract_news(company_name, num_articles)

        if not news_articles.get("Articles"):
            raise HTTPException(status_code=404, detail="No articles found.")

        # Run comparative analysis
        updated_news_articles = generate_comparative_analysis(news_articles)

        # Add topic overlap analysis
        updated_news_articles = add_topic_overlap_to_coverage_differences(
            updated_news_articles
        )

        # Generate company analysis in Hindi
        sentiment_report = generate_company_analysis_in_hindi(updated_news_articles)

        # Convert analysis to speech
        hindi_audio = convert_text_to_speech(sentiment_report, "output.mp3")
        updated_news_articles["Audio"] = hindi_audio

        return updated_news_articles

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
