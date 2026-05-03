import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import (
    MarianMTModel, MarianTokenizer,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration, MBart50TokenizerFast
)

st.set_page_config(
    page_title="Translation Model Comparison Dashboard",
    layout="wide"
)

df = pd.read_excel("final_comparison.xlsx")

summary = pd.DataFrame({
    "Model": ["TranslateGemma", "MarianMT", "NLLB", "mBART"],
    "Average Cosine Similarity": [
        df["sim_gemma"].mean(),
        df["sim_marian"].mean(),
        df["sim_nllb"].mean(),
        df["sim_mbart"].mean()
    ],
    "Accuracy (%)": [
        df["acc_gemma"].mean() * 100,
        df["acc_marian"].mean() * 100,
        df["acc_nllb"].mean() * 100,
        df["acc_mbart"].mean() * 100
    ]
})

@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity(text1, text2):
    sim_model = load_similarity_model()
    emb1 = sim_model.encode(str(text1))
    emb2 = sim_model.encode(str(text2))
    return cosine_similarity([emb1], [emb2])[0][0]

@st.cache_resource
def load_marian_en_hi():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_marian_hi_en():
    model_name = "Helsinki-NLP/opus-mt-hi-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def marian_translate(text, direction):
    if direction == "en-hi":
        tokenizer, model = load_marian_en_hi()
    else:
        tokenizer, model = load_marian_hi_en()

    inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@st.cache_resource
def load_nllb():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def nllb_translate(text, src_lang, tgt_lang):
    tokenizer, model = load_nllb()
    tokenizer.src_lang = src_lang

    inputs = tokenizer(str(text), return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@st.cache_resource
def load_mbart():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def mbart_translate(text, src_lang, tgt_lang):
    tokenizer, model = load_mbart()
    tokenizer.src_lang = src_lang

    inputs = tokenizer(str(text), return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
    )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


st.title("Translation Model Comparison Dashboard")
st.caption("Compare translation models using back-translation, cosine similarity, and threshold-based accuracy.")

top_left, top_right = st.columns([2, 1])

with top_left:
    st.subheader("Model Results Preview")
    st.dataframe(df.head())

with top_right:
    st.subheader("Try Translating!!")

    user_sentence = st.text_input("Enter any English sentence")
    live_model = st.selectbox(
        "Select translation model",
        ["MarianMT", "NLLB", "mBART"]
      
    )
    st.caption("Live demo supports MarianMT, NLLB, and mBART. TranslateGemma was evaluated separately using Ollama.")

    if st.button("Translate"):
        if user_sentence.strip() == "":
            st.warning("Please enter a sentence.")
        else:
            with st.spinner("Translating..."):
                if live_model == "MarianMT":
                    hindi = marian_translate(user_sentence, "en-hi")
                    back = marian_translate(hindi, "hi-en")

                elif live_model == "NLLB":
                    hindi = nllb_translate(user_sentence, "eng_Latn", "hin_Deva")
                    back = nllb_translate(hindi, "hin_Deva", "eng_Latn")

                else:
                    hindi = mbart_translate(user_sentence, "en_XX", "hi_IN")
                    back = mbart_translate(hindi, "hi_IN", "en_XX")

                sim_score = compute_similarity(user_sentence, back)

            st.write("### Output")
            st.write("**Hindi Translation:**", hindi)
            st.write("**Back Translation:**", back)
            st.write("**Cosine Similarity:**", round(sim_score, 4))

st.divider()

st.subheader("Results Summary")
st.dataframe(summary)

st.subheader("Accuracy Comparison")
accuracy_data = summary[["Model", "Accuracy (%)"]].set_index("Model")
st.bar_chart(accuracy_data)

st.subheader("Average Cosine Similarity Comparison")
cosine_data = summary[["Model", "Average Cosine Similarity"]].set_index("Model")
st.bar_chart(cosine_data)

st.subheader("Worst Translation Cases")

model_choice = st.selectbox(
    "Select model for worst-case analysis",
    ["Gemma", "MarianMT", "NLLB", "mBART"]
)

sim_col_map = {
    "Gemma": "sim_gemma",
    "MarianMT": "sim_marian",
    "NLLB": "sim_nllb",
    "mBART": "sim_mbart"
}

back_col_map = {
    "Gemma": "back_gemma",
    "MarianMT": "back_marian",
    "NLLB": "back_nllb",
    "mBART": "back_mbart"
}

sim_col = sim_col_map[model_choice]
back_col = back_col_map[model_choice]

worst_cases = df.sort_values(by=sim_col).head(10)

st.dataframe(
    worst_cases[[
        "sentence_id",
        "en_original",
        back_col,
        sim_col
    ]]
)

best_model = summary.loc[
    summary["Average Cosine Similarity"].idxmax(),
    "Model"
]

st.success(f"Best performing model based on average cosine similarity: {best_model}")