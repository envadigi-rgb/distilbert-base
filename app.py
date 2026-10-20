import gradio as gr
from transformers import pipeline

nlp = pipeline("sentiment-analysis",
               model="distilbert-base-uncased-finetuned-sst-2-english",
               device=-1)


def run(text: str) -> str:
    if not text.strip():
        return "Please enter some text."
    r = nlp(text)[0]
    return f"{r['label']} ({r['score']:.4f})"


demo = gr.Interface(fn=run,
                    inputs=gr.Textbox(label="Text", placeholder="I love this!"),
                    outputs=gr.Textbox(label="Sentiment"),
                    title="Shipwright demo")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
