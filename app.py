import os
import re
import html
import gradio as gr
import google.generativeai as genai
from google.generativeai.types import content_types

# === Load Gemini API Key ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
if not GEMINI_API_KEY:
    raise ValueError("⚠️ Set GEMINI_API_KEY in environment or Hugging Face Secrets.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# === Helper Functions ===

def gemini_generate(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini API Error: {e}"

def analyze_text(text: str) -> str:
    prompt = (
        "Analyse the following text. "
        "Return sentiment (positive, negative, or neutral), topics, word count, and readability score.\n\n"
        f"{text}"
    )
    return gemini_generate(prompt)

def parse_analysis_output(response: str):
    sentiment = ""
    topics = ""
    word_count = ""
    readability = ""

    try:
        sentiment_match = re.search(r"\*\*Sentiment:\*\*\s*(.*)", response)
        topics_match = re.search(r"\*\*Topics:\*\*\s*(.*)", response)
        wordcount_match = re.search(r"\*\*Word Count:\*\*\s*(.*)", response)
        readability_match = re.search(r"\*\*Readability Score:\*\*\s*(.*)", response, re.DOTALL)

        sentiment = sentiment_match.group(1).strip() if sentiment_match else ""
        topics = topics_match.group(1).strip() if topics_match else ""
        word_count = wordcount_match.group(1).strip() if wordcount_match else ""
        readability = readability_match.group(1).strip() if readability_match else response.strip()
    except Exception as e:
        print(f"⚠️ Parsing error: {e}")
        readability = response.strip()

    return sentiment, topics, word_count, readability
    
def analyze_text_split_output(text):
    response = analyze_text(text)
    return parse_analysis_output(response)

def summarize_text(text: str) -> str:
    prompt = f"Summarise this text in 2-3 sentences:\n\n{text}"
    return gemini_generate(prompt)

def process_document(file):
    yield "⏳ *Processing document...*"

    if file is None:
        yield "❗ **No file uploaded.**"
        return

    ext = os.path.splitext(file.name)[-1].lower()
    try:
        if ext == ".pdf":
            try:
                uploaded_file = genai.upload_file(file.name, mime_type="application/pdf")
                response = model.generate_content(
                    ["Summarise this document:", uploaded_file]
                )
                text = response.text.strip()
            except Exception as e:
                yield f"⚠️ **Could not summarise PDF:** {e}"
                return
        elif ext == ".txt":
            try:
                # Try UTF-8 first
                with open(file.name, "r", encoding="utf-8") as f:
                    input_text = f.read()
            except UnicodeDecodeError:
                try:
                    # Try fallback encoding
                    with open(file.name, "r", encoding="ISO-8859-1") as f:
                        input_text = f.read()
                except Exception as e:
                    yield f"⚠️ **Could not read TXT:** {e}"
                    return
        
            text = summarize_text(input_text)
            text = html.escape(text)
        else:
            yield "⚠️ **Only PDF and TXT files are supported.**"
            return
    except Exception as e:
        yield f"⚠️ **Could not summarise:** {e}"
        return

    # Wrap the output in a scrollable div
    yield text

# ===== UI Title and Description =====
APP_TITLE = "🧘‍♂️ ManoVākya (मनॊवाक्य): Sentiments & Summaries"
APP_DESCRIPTION = (
    "Understand your thoughts, messages, or documents through the lens of AI. "
    "ManoVākya helps analyse tone, extract meaning, and distil essence — enabling clarity in daily communication, journaling, and decision-making."
)

# ===== Gradio Interfaces =====
analyze_interface = gr.Interface(
    fn=analyze_text_split_output,
    inputs=gr.Textbox(lines=5, max_lines=20, label="🗣 Enter text for sentiment & topic analysis"),
    outputs=[
        gr.Textbox(label="📊 Sentiment", lines = 1),
        gr.Textbox(label="🧩 Topics", lines = 2, max_lines=20),
        gr.Textbox(label="🔢 Word Count", lines = 1),
        gr.Textbox(label="📘 Readability Score", lines = 4, max_lines=20)
    ],
    title="Gemini Sentiment & Topic Analyzer",
    flagging_mode="never",
    live=False
)

summarize_interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=5, max_lines=20, label="✍️ Enter text to summarise"),
    outputs=gr.Textbox(label="📝 Summary Result", lines = 4, max_lines=20),
    title="Gemini Text Summariser",
    flagging_mode="never",
    live=False
)

#doc_interface = gr.Interface(
#    fn=process_document,
#    inputs=gr.File(label="📁 Upload PDF or TXT File"),
#    outputs=gr.Markdown(label="📝 Summary"),
#    title="Gemini Document Summariser",
#    flagging_mode="never",
#    live=True
#)

with gr.Blocks(title="Gemini Document Summariser") as doc_interface:
    gr.Markdown("## 📁 Gemini Document Summariser")

    file_input = gr.File(label="Upload PDF or TXT File")
    submit_button = gr.Button("Submit")
    output_box = gr.Markdown(label="📝 Summary")

    submit_button.click(
        fn=process_document,
        inputs=file_input,
        outputs=output_box
    )
    
# Try to load the PayPal URL from the environment; if missing, use a placeholder
paypal_url = os.getenv("PAYPAL_URL", "https://www.paypal.com/donate/dummy-link")

# ===== Launch Tabbed UI =====
with gr.Blocks(title=APP_TITLE) as the_application:
    gr.HTML(f"""
        <div style='text-align: center; margin-bottom: 20px;'>
            <p style='font-size: 40px; font-weight: bold;'>{APP_TITLE}</p>
            <p style='font-size: 16px; line-height: 1.6;'>{APP_DESCRIPTION}</p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🗣 Analyse Text"):
            analyze_interface.render()
        with gr.TabItem("✍️ Summarise Text"):
            summarize_interface.render()
        with gr.TabItem("📁 Summarise Document"):
            doc_interface.render()

    with gr.Row():
        gr.HTML(f"""
        <a href="{paypal_url}" target="_blank">
            <button style="background-color:#0070ba;color:white;border:none;padding:10px 20px;
            font-size:16px;border-radius:5px;cursor:pointer;margin-top:10px;">
                ❤️ Support Research via PayPal
            </button>
        </a>
        """)
        
# Determine if running on Hugging Face Spaces
on_spaces = os.environ.get("SPACE_ID") is not None

the_application.launch(share=not on_spaces)