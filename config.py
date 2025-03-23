from huggingface_hub import InferenceClient
from langchain_groq import ChatGroq
from google import genai
from openai import OpenAI
from translate import Translator
import os

# Read from secrets
API_URL = "https://router.huggingface.co/hf-inference/models/facebook/mms-tts-hin"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# Initialize OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Initialize the translator
translator = Translator(to_lang="hi")

# Initialize GenAI clients
genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
genai_client2 = genai.Client(api_key=os.getenv("GEMINI_API_KEY_2"))

# Initialize the InferenceClient
client = InferenceClient(provider="hf-inference", api_key=os.getenv("HF_API_KEY"))

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=None,
    timeout=None,
)
