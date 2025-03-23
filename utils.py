# utils.py
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import time
from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
import os
from google import genai
from translate import Translator
import re
from openai import OpenAI
import json
from config import (
    llm,
    client,
    API_URL,
    HEADERS,
    openrouter_client,
    translator,
    genai_client,
    genai_client2,
)


def extract_news(company, num_articles=10):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }

    articles = []
    seen_urls = set()
    sentiment_count = {"Positive": 0, "Negative": 0, "Neutral": 0}

    print(f"Fetching {num_articles} valid articles...")

    while len(articles) < num_articles:
        try:
            for url in search(f"{company} news", num_results=num_articles * 2):
                if url in seen_urls or any(
                    blocked in url
                    for blocked in ["youtube.com", "youtu.be", "twitter.com", "x.com"]
                ):
                    continue

                seen_urls.add(url)
                print(f"\nFetching article {len(articles) + 1}: {url}")

                try:
                    article_response = requests.get(url, headers=headers, timeout=10)
                    article_response.raise_for_status()

                    article_soup = BeautifulSoup(article_response.text, "html.parser")
                    title = (
                        article_soup.find("title").get_text()
                        if article_soup.find("title")
                        else "No Title Found"
                    )

                    if title == "No Title Found":
                        print(f"Skipping article {url} due to missing title")
                        continue

                    paragraphs = article_soup.find_all("p")
                    content = " ".join([para.get_text() for para in paragraphs])

                    if not content.strip():
                        print(f"No content found for {url}")
                        continue

                    # Summarize the article (Use existing summarize function)
                    summary, translated_summary = summarize_article(content)

                    # Sentiment analysis
                    sentiment, reason = analyze_sentiment(summary)
                    if sentiment in sentiment_count:
                        sentiment_count[sentiment] += 1

                    # Extract relevant topics using LLM
                    topics = extract_topics(content)

                    metadata = {
                        "URL": url,
                        "Title": title,
                        "Summary": summary,
                        "Summary(in Hindi)": translated_summary,
                        "Sentiment": sentiment,
                        "Reason(for Sentiment)": reason,
                        "Topics": topics,
                    }
                    articles.append(metadata)

                    if len(articles) >= num_articles:
                        break

                    time.sleep(1)

                except requests.exceptions.Timeout:
                    print(f"Timeout error while fetching {url}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch article {url}: {e}")

        except Exception as e:
            print(f"Error while searching for articles: {e}")
    final_sentiment = final_sentiment_analysis(sentiment_count)
    # Create final JSON output
    output = {
        "Company": company,
        "Articles": articles,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_count,
            "Total Sentiments": sum(sentiment_count.values()),
        },
        "Final Sentiment Analysis": final_sentiment,
    }

    print(f"\nSuccessfully fetched {len(articles)} valid articles.")

    # Pretty print JSON
    print(json.dumps(output, indent=4))
    return output


def clean_and_parse_json(response):
    try:
        # Extract JSON block using regex
        cleaned_response = re.search(r"\{.*\}", response, re.DOTALL)
        if cleaned_response:
            cleaned_response = cleaned_response.group()

            # Replace single quotes with double quotes
            cleaned_response = cleaned_response.replace("'", '"')

            # Remove trailing commas
            cleaned_response = re.sub(r",\s*([\]}])", r"\1", cleaned_response)

            # Parse JSON
            return json.loads(cleaned_response)
        else:
            raise ValueError("No JSON found in response.")
    except Exception as e:
        print(f"Parsing error: {e}")
        return {"Analysis": "Failed to parse structured output from the model."}


def final_sentiment_analysis(sentiment_count):
    try:
        sentiment_data = json.dumps(sentiment_count, indent=2)
        print("Sentiment Data:\n", sentiment_data)

        # Use Gemma for sentiment analysis
        response = genai_client2.models.generate_content(
            model="gemma-3-27b-it",
            contents=(
                f"Analyze the following sentiment distribution:\n\n"
                f"{sentiment_data}\n\n"
                f"Provide a FINAL sentiment analysis based on the distribution. "
                f"Summarize the overall sentiment and any patterns or insights in EXACTLY THREE LINES. "
                f"Return the result in a normal string format with summary about the sentiment and particular company current form."
            ),
        )

        if response and response.text:
            generated_text = response.text.strip()
            print("Generated Text (Gemma):", generated_text)
        else:
            print("Error: No response from the model.")
            generated_text = "Failed to generate response from Gemma."
            print("Generated Text:", generated_text)

        # Clean and parse JSON
        # generated_text = clean_and_parse_json(generated_text)
        return generated_text

    except Exception as e:
        print(f"Gemma sentiment analysis failed: {e}")
        try:
            print("Falling back to OpenRouter (qwen/qwq)...")
            completion = openrouter_client.chat.completions.create(
                model="qwen/qwq-32b:free",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Analyze the following sentiment distribution:\n\n"
                            f"{sentiment_data}\n\n"
                            f"Provide a FINAL sentiment analysis based on the distribution. "
                            f"Summarize the overall sentiment and any patterns or insights in EXACTLY THREE LINES. "
                            f"Return the result in a normal string format with summary about the sentiment and particular company current form."
                        ),
                    }
                ],
            )

            generated_text = completion.choices[0].message.content.strip()
            print("OpenRouter Generated Text:", generated_text)

            # Clean and parse JSON
            # generated_text = clean_and_parse_json(generated_text)
            return generated_text

        except Exception as e:
            print(f"OpenRouter sentiment analysis failed: {e}")
            return {
                "Summary": "Sentiment analysis failed using both models.",
                "Dominant Sentiment": "Unknown",
                "Insight": "No insights available.",
            }


def summarize_article(content):
    try:
        # Use the Inference API for summarization
        result = client.summarization(content, model="facebook/bart-large-cnn")
        summary = result["summary_text"]

        # Translate summary to Hindi
        translated_summary = translator.translate(summary)

        return summary, translated_summary

    except Exception as e:
        print(f"Hugging Face summarization failed: {e}")
        print("Trying Gemini as fallback...")

        try:
            truncated_content = content[:1024]
            # Use Gemini model as fallback
            response = genai_client2.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"Summarize the following article:\n\n{truncated_content}",
            )
            summary = response.text.strip()
            translated_summary = translator.translate(summary)

            return summary, translated_summary

        except Exception as e:
            print(f"Gemini summarization also failed: {e}")
            return "Summary not available", "Summary not available"


def convert_text_to_speech(text, filename):
    try:
        # Try using gTTS for Hindi TTS
        tts = gTTS(text=text, lang="hi")
        tts.save(filename)

        return filename

    except Exception as e:
        print(f"Error during TTS conversion with gTTS: {e}")
        print("Switching to Hugging Face MMS-TTS...")

        # Use Hugging Face MMS-TTS as a fallback
        try:
            payload = {"inputs": text}
            response = requests.post(API_URL, headers=HEADERS, json=payload)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                print("Audio file generated using Hugging Face MMS-TTS.")
                return filename
            else:
                print(
                    f"Hugging Face API error: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            print(f"Error during TTS conversion with Hugging Face MMS-TTS: {e}")
            return None


def analyze_sentiment(content):
    try:
        # Use Gemini LLM for sentiment analysis as the primary method
        truncated_content = content
        response = genai_client.models.generate_content(
            model="gemma-3-27b-it",
            contents=(
                f"Analyze the sentiment of the following article:\n\n{truncated_content}\n\n"
                "Return the response as simple string . It shoild have sentiment and reason both are seperated by comma"
                "Sentiment should be Positive, Negative, or Neutral. No other extra text or contents."
            ),
        )
        # print("RESponse", response)
        # Parse JSON response
        # generated_text = response.text.strip()
        # print("Generated Response:", generated_text)

        # Split the response into sentiment and reason
        parts = response.text.split(",", 1)
        if len(parts) == 2:
            sentiment = parts[0].strip()
            reason = parts[1].strip()
            print(f"Sentiment: {sentiment}")
            print(f"Reason: {reason}")
        else:
            print("Invalid response format")

        if sentiment.lower() not in ["positive", "negative", "neutral"]:
            sentiment = "Sentiment analysis not available"

        return sentiment, reason

    except Exception as e:
        print(f"Error during Gemini LLM sentiment analysis: {e}")
        print("Trying HF Inference as fallback for sentiment analysis...")

        try:
            # Fallback to HF Inference for sentiment analysis
            truncated_content = content
            result = client.text_classification(
                truncated_content,
                model="finiteautomata/bertweet-base-sentiment-analysis",
            )
            hf_sentiment = result[0]["label"]

            # Map HF inference sentiment labels to required format
            sentiment_mapping = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}

            sentiment = sentiment_mapping.get(
                hf_sentiment, "Sentiment analysis not available"
            )
            return sentiment, ""

        except Exception as e:
            print(f"HF Inference sentiment analysis also failed: {e}")
            return "Sentiment analysis not available", ""


def extract_topics(content):
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that extracts key topics from articles.",
                ),
                (
                    "human",
                    f"Extract the **three most relevant and suitable two words topics** from the following article:\n\n{content}\n\n"
                    "Provide exactly 5 topics, separated by commas. Avoid numbering or extra details.",
                ),
            ]
        )
        chain = prompt | llm
        response = chain.invoke({})

        # Clean and process response
        if isinstance(response, str):
            topics = response.split(",")
        elif hasattr(response, "content") and isinstance(response.content, str):
            topics = response.content.split(",")
        else:
            topics = ["Topics not available"]

        # Remove unwanted characters and extra spaces
        topics = [re.sub(r"^\d+[\.\)\-]?\s*", "", topic.strip()) for topic in topics]

        return topics

    except Exception as e:
        print(f"Groq topic extraction failed: {e}")
        print("Trying Gemini as fallback...")

        try:
            # Use Gemini for topic extraction
            response = genai_client.models.generate_content(
                model="gemma-3-27b-it",
                contents=f"Extract exactly 5 key topics (two or one words) from the following article:\n\n{content}\n\n"
                "Return them as a comma-separated list without numbering or extra details.",
            )
            topics = response.text.strip().split(",")

            # Clean up and format the topics
            topics = [
                re.sub(r"^\d+[\.\)\-]?\s*", "", topic.strip()) for topic in topics
            ]

            return topics

        except Exception as e:
            print(f"Gemini topic extraction also failed: {e}")
            return ["Topics not available"]


def clean_response(response):
    cleaned_response = re.search(r"\{.*\}", response, re.DOTALL)
    if cleaned_response:
        return cleaned_response.group()
    return response


def generate_comparative_analysis(articles):
    num_articles = min(len(articles["Articles"]), 5)

    all_contents = "\n\n".join(
        [
            f"Article {i + 1}: {article['Summary']}"
            for i, article in enumerate(articles["Articles"][:num_articles])
        ]
    )

    print(all_contents)

    try:
        # Use Gemma for comparative analysis
        response = genai_client.models.generate_content(
            model="gemma-3-27b-it",
            contents=(
                f"Perform a detailed comparative analysis of the following {num_articles} articles:\n\n"
                f"{all_contents}\n\n"
                f"Provide differences in coverage and potential impacts between each pair of articles.\n"
                f"Return the result in the following structured JSON format:\n"
                f"{{\n"
                f'  "Coverage Differences": [\n'
                f"    {{\n"
                f'      "Comparison": "<summary of difference between article X and article Y>",\n'
                f'      "Impact": "<impact of these differences>"\n'
                f"    }}\n"
                f"  ]\n"
                f"}}"
            ),
        )

        generated_text = response.text.strip()
        print("Generated Text (Gemma):", generated_text)

        # Clean and parse the response
        cleaned_response = clean_response(generated_text)
        structured_response = json.loads(cleaned_response)

    except Exception as e:
        print(f"Gemma comparative analysis failed: {e}")

        try:
            print("Falling back to OpenRouter (qwq)...")
            if "all_contents" not in locals():
                all_contents = "\n\n".join(
                    [
                        f"Article {i+1}: {article['Summary']}"
                        for i, article in enumerate(articles["Articles"])
                    ]
                )
            completion = openrouter_client.chat.completions.create(
                model="qwen/qwq-32b:free",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Perform a detailed comparative analysis of the following {num_articles} articles:\n\n"
                            f"{all_contents}\n\n"
                            f"Provide differences in coverage and potential impacts between each pair of articles.\n"
                            f"Return the result in the following structured JSON format:\n"
                            f"{{\n"
                            f'  "Coverage Differences": [\n'
                            f"    {{\n"
                            f'      "Comparison": "<summary of difference between article X and article Y>",\n'
                            f'      "Impact": "<impact of these differences>"\n'
                            f"    }}\n"
                            f"  ]\n"
                            f"}}"
                        ),
                    }
                ],
            )

            # Extract response
            generated_text = completion.choices[0].message.content.strip()
            print("Generated Text (OpenRouter):", generated_text)

            # Clean and parse the response
            cleaned_response = clean_response(generated_text)
            structured_response = json.loads(cleaned_response)

        except Exception as e:
            print(f"OpenRouter fallback failed: {e}")
            structured_response = {
                "Coverage Differences": [
                    {
                        "Comparison": "Failed to generate comparative analysis from both Gemma and OpenRouter.",
                        "Impact": "Check the model's output format or prompt.",
                    }
                ]
            }

    #  Merge the result back into the input
    articles["Coverage Differences"] = structured_response["Coverage Differences"]
    return articles


def add_topic_overlap_to_coverage_differences(news_articles):
    try:
        # Get the list of topics from each article
        article_topics = [
            set(article["Topics"]) for article in news_articles["Articles"]
        ]

        if len(article_topics) < 2:
            raise ValueError(
                "At least two articles are required for topic overlap analysis."
            )

        # Step 1: Find topics that are common in at least two articles
        topic_occurrences = {}
        for i, topics in enumerate(article_topics):
            for topic in topics:
                if topic not in topic_occurrences:
                    topic_occurrences[topic] = []
                topic_occurrences[topic].append(
                    i + 1
                )  # Storing article index (1-based)

        # Step 2: Filter out topics appearing in at least two articles
        common_topics = [
            f"{topic} (Article {', '.join(map(str, indexes))})"
            for topic, indexes in topic_occurrences.items()
            if len(indexes) >= 2
        ]

        # Step 3: Find unique topics for each article
        unique_topics = []
        for i, topics in enumerate(article_topics):
            other_topics = set.union(*(article_topics[:i] + article_topics[i + 1 :]))
            unique = topics - other_topics
            unique_topics.append(list(unique))

        # Step 4: Build the output structure for Coverage Differences
        topic_overlap = {"Common Topics": common_topics, "Unique Topics": {}}

        for i, topics in enumerate(unique_topics):
            topic_overlap["Unique Topics"][f"Article {i + 1}"] = topics

        #  Step 5: Add topic_overlap under "Coverage Differences"
        if "Coverage Differences" not in news_articles:
            news_articles["Coverage Differences"] = []

        news_articles["Coverage Differences"].append({"Topic Overlap": topic_overlap})

        return news_articles

    except Exception as e:
        print(f"Error: {e}")
        return None


def generate_company_analysis_in_hindi(company_name, articles):
    try:
        # Extract summaries and sentiment from updated articles
        summaries = "\n\n".join(
            [
                f"Article {i+1}: {article['Summary']}\nSentiment: {article.get('Sentiment', 'Unknown')}"
                for i, article in enumerate(articles["Articles"])
            ]
        )

        # Create the prompt in English, request output in Hindi
        prompt = (
            f"Please prepare a detailed analysis of the company '{company_name}' based on the following articles. "
            f"Include the sentiment and key points from each article.\n\n"
            f"{summaries}\n\n"
            f"The analysis should cover the overall performance of the company, the sentiment reflected in the articles, "
            f"and a comparison of the articles' coverage.\n\n"
            f"Output should be in **Hindi** and in a single detailed paragraph based on these articles.Don't provide in english"
        )

        # Generate content using Gemini Gemma model
        response = genai_client2.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt,
        )

        if response and response.text:
            analysis_text = response.text.strip()
            print("\n🔎 Generated Company Analysis (Hindi):\n")
            print(analysis_text)
            return analysis_text
        else:
            raise ValueError("Gemini returned an empty response")

    except Exception as e:
        print(f"Gemini analysis generation failed: {e}")
        return None
