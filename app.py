import streamlit as st
from utils import (
    extract_news,
    generate_comparative_analysis,
    add_topic_overlap_to_coverage_differences,
    generate_company_analysis_in_hindi,
    convert_text_to_speech,
)
import json


# Streamlit app
def main():
    st.title("Company News Analyzer")

    # Input for company name
    company_name = st.text_input("Enter company name:")

    if company_name:
        st.write(f"Fetching news for {company_name}...")

        # Extract news articles
        news_articles = extract_news(company_name)

        if news_articles.get("Articles"):

            # Generate comparative analysis
            updated_news_articles = generate_comparative_analysis(news_articles)

            # Add topic overlap analysis
            updated_news_articles = add_topic_overlap_to_coverage_differences(
                updated_news_articles
            )

            # Generate company analysis in Hindi
            sentiment_report = generate_company_analysis_in_hindi(
                company_name, updated_news_articles
            )

            if sentiment_report:
                # Display in sidebar
                st.sidebar.write("**Company Analysis in Hindi:**")
                st.sidebar.write(sentiment_report)

                # Convert text to speech
                hindi_audio = convert_text_to_speech(sentiment_report, "output.mp3")

                if hindi_audio:
                    st.audio(hindi_audio, format="audio/mp3")

            st.write("Updated Articles with Comparative Analysis and Topic Overlap:")
            st.json(updated_news_articles)

        else:
            st.write("No articles found.")


if __name__ == "__main__":
    main()
