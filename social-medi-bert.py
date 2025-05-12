mport streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Başlık ve açıklama
st.title("Content Relevance Classifier")
st.write("This app checks the semantic similarity between the given text and a selected content category using a BERT model.")

# BERT modeli
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

bert_model = load_model()

# Kategori seçenekleri
content_options = [
    "People", "Law", "Cryptocurrency", "Politics", "Science",
    "Technology", "Business", "Entertainment", "Investing", "Finance",
    "Environment", "Social", "Economy", "Sports", "Health"
]

# Kullanıcı girişi
text = st.text_area("Enter your text:", height=200)
category = st.selectbox("Select a content category:", content_options)

# Tahmin butonu
if st.button("Check Relevance"):
    if text and category:
        text_embed = bert_model.encode([text])
        category_embed = bert_model.encode([category])
        similarity = cosine_similarity(text_embed, category_embed)[0][0]
        st.write(f"🔍 Similarity Score: {similarity:.3f}")

        result = "✅ Relevant" if similarity >= 0.10 else "❌ Non-relevant"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter text and select a category.")
