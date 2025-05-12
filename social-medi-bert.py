import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

model = load_model()

# Dataset bağlantısı
DATASET_URL = "https://huggingface.co/datasets/eda12/social-relevance-data/resolve/main/processed_data.csv"

# Arayüz
st.title("🔍 Content Relevance Classifier")
st.write("This model checks the semantic similarity between input text and the selected content category using BERT.")

# Kategoriler
content_options = [
    "People", "Law", "Cryptocurrency", "Politics", "Science",
    "Technology", "Business", "Entertainment", "Investing", "Finance",
    "Environment", "Social", "Economy", "Sports", "Health"
]

# Kullanıcı girişleri
text = st.text_area("Enter your text:", height=200)
category = st.selectbox("Select content category:", content_options)

# Dataseti göster (isteğe bağlı)
with st.expander("📊 View Sample from Dataset"):
    df = pd.read_csv(DATASET_URL)
    st.dataframe(df.head())

# Hesapla
if st.button("Check Relevance"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        text_embed = model.encode([text])
        category_embed = model.encode([category])
        similarity = cosine_similarity(text_embed, category_embed)[0][0]

        st.markdown(f"**Similarity Score:** `{similarity:.3f}`")

        if similarity >= 0.10:
            st.success("✅ Relevant")
        else:
            st.error("❌ Non-Relevant")
