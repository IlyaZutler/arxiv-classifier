import json
import torch
import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Загрузка модели (кэшируется после первого запуска) ──────────────────────
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("model/")
    model = DistilBertForSequenceClassification.from_pretrained("model/")
    model.eval()

    with open("model/label2class.json") as f:
        label2class = json.load(f)  # {"0": "cs.AI", ...}

    with open("taxonomy_clean.json") as f:
        tax = json.load(f)

    return tokenizer, model, label2class, tax["sub"], tax["top"]

# ── Предсказание ────────────────────────────────────────────────────────────
def predict(title, abstract, tokenizer, model, label2class, sub_taxonomy, top_names):
    text = title.strip() + " [SEP] " + abstract.strip()
    enc = tokenizer(
        text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)

    # Сортируем по убыванию вероятности
    sorted_idx = probs.argsort(descending=True).tolist()

    # Топ-95%: берём классы пока накопленная вероятность < 0.95
    results = []
    cumsum = 0.0
    for idx in sorted_idx:
        code = label2class[str(idx)]          # "cs.LG"
        prob = probs[idx].item()

        # Человекочитаемое название
        if code in sub_taxonomy:
            name = f"{sub_taxonomy[code]} ({code})"
        elif code in top_names:
            name = f"{top_names[code]} ({code})"
        else:
            name = code

        results.append((name, prob))
        cumsum += prob
        if cumsum >= 0.95:
            break

    return results

# ── Интерфейс ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ArXiv Classifier", page_icon="📄")
st.title("📄 ArXiv Paper Classifier")
st.caption("Введите название и/или аннотацию статьи — модель определит тематику")

tokenizer, model, label2class, sub_taxonomy, top_names = load_model()

title    = st.text_input("Название статьи (Title)", placeholder="например: Attention Is All You Need")
abstract = st.text_area("Аннотация (Abstract)", placeholder="We propose a new architecture...", height=200)

if st.button("Определить тематику", type="primary"):
    if not title.strip() and not abstract.strip():
        st.warning("Введите хотя бы название или аннотацию")
    else:
        with st.spinner("Классифицирую..."):
            results = predict(title, abstract, tokenizer, model, label2class, sub_taxonomy, top_names)

        st.subheader("Топ-95% тематик:")
        for name, prob in results:
            st.progress(prob, text=f"**{name}** — {prob:.1%}")
