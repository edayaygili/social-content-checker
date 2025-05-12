import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
from io import StringIO

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

model = load_model()

# Veri setini HuggingFace URL'sinden yükle
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/eda12/social-relevance-data/resolve/main/processed_data.csv"
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    return data

df = load_data()

# Sayfa başlığı
st.title("🔍 AI Content Relevance Checker")
st.markdown("BERT modeliyle context ve kategori uyumunu değerlendirir.")

# Dataset örneği göster
with st.expander("📂 Show dataset sample"):
    st.dataframe(df.head())

# Kullanıcıdan giriş al
context = st.text_area("📝 Context (Text)", height=200)
unique_categories = sorted(df['content'].dropna().unique())
category = st.selectbox("📚 Content Category", unique_categories)

# Relevance kontrolü
if st.button("Check Relevance"):
    if context and category:
        context_embed = model.encode([context])
        category_embed = model.encode([category])
        similarity = cosine_similarity(context_embed, category_embed)[0][0]

        st.markdown(f"**🔗 Similarity Score:** `{similarity:.3f}`")

        if similarity >= 0.10:
            st.success("✅ Relevant")
        else:
            st.error("❌ Non-Relevant")
    else:
        st.warning("Please provide both context and category.")
