import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

model = load_model()

# Dataseti yükle
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/eda12/social-relevance-data/resolve/main/processed_data.csv"
    return pd.read_csv(url)

df = load_data()

# Arayüz başlığı
st.title("🔍 AI Content Relevance Checker")
st.markdown("BERT modeliyle context ve kategori uyumunu değerlendirir.")

# Kullanıcı veri seçebilir
st.subheader("1️⃣ Örnek veri seç (isteğe bağlı)")
row_index = st.slider("Veri Satırı Seç (0 - {})".format(len(df)-1), 0, len(df)-1, 0)
sample_text = df.loc[row_index, "text"]
sample_category = df.loc[row_index, "content"]

st.text_area("Context (Text)", value=sample_text, height=200, key="context")
selected_category = st.selectbox(
    "Content Category",
    [
        "People", "Law", "Cryptocurrency", "Politics", "Science",
        "Technology", "Business", "Entertainment", "Investing", "Finance",
        "Environment", "Social", "Economy", "Sports", "Health"
    ],
    index=max(0, content_options.index(sample_category)) if 'sample_category' in locals() else 0
)

# Tahmin
if st.button("Check Relevance"):
    context_embed = model.encode([st.session_state.context])
    category_embed = model.encode([selected_category])
    similarity = cosine_similarity(context_embed, category_embed)[0][0]

    st.markdown(f"**Similarity Score:** {similarity:.3f}")

    if similarity >= 0.10:
        st.success("✅ Relevant")
    else:
        st.error("❌ Non-Relevant")
