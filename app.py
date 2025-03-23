import streamlit as st
import requests
import json
import os

API_URL = "http://127.0.0.1:8000/extract-news/"


def main():
    st.title("Company News Analyzer")

    # Input for company name
    company_name = st.text_input("Enter company name:")

    # Add a Submit button
    if st.button("Submit"):
        if company_name:
            st.write(f"Fetching news for {company_name}...")

            try:
                response = requests.post(API_URL, json={"company_name": company_name})

                if response.status_code == 200:
                    data = response.json()

                    # Display sentiment report in sidebar
                    if "sentiment_report" in data and data["sentiment_report"]:
                        st.sidebar.write("**Company Analysis in Hindi:**")
                        st.sidebar.write(data["sentiment_report"])

                        # Play audio file if available
                        if data.get("audio_file") and os.path.exists(
                            data["audio_file"]
                        ):
                            with open(data["audio_file"], "rb") as audio_file:
                                st.audio(audio_file, format="audio/mp3")

                    # Display comparative analysis and topic overlap
                    if "articles" in data:
                        st.write(
                            "Updated Articles with Comparative Analysis and Topic Overlap:"
                        )
                        st.json(data["articles"])

                else:
                    st.write(f"Error: {response.json().get('detail')}")

            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Please enter a company name.")


if __name__ == "__main__":
    main()
