from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# BERT modeli
bert_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# Güncel kategoriler
content_options = [
    "People", "Law", "Cryptocurrency", "Politics", "Science",
    "Technology", "Business", "Entertainment", "Investing", "Finance",
    "Environment", "Social", "Economy", "Sports", "Health"
]

# Tahmin fonksiyonu
def semantic_predict(text, category):
    text_embed = bert_model.encode([text])
    category_embed = bert_model.encode([category])
    similarity = cosine_similarity(text_embed, category_embed)[0][0]
    print(f"🔍 Similarity: {similarity:.3f}")

    return "relevant" if similarity >= 0.10 else "non-relevant"

# Gradio arayüzü
gr.Interface(
    fn=semantic_predict,
    inputs=[
        gr.Textbox(lines=6, label="Text"),
        gr.Dropdown(choices=content_options, label="Content Category")
    ],
    outputs=gr.Textbox(label="Predicted Relevance"),
    title="Content Relevance Classifier",
    description="This model checks semantic similarity between the text and selected content category."
).launch(share=True)

