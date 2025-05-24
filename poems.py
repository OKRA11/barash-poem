import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Интерфейс
st.title("🎤 Бараш пишет стихи")
st.image("https://upload.wikimedia.org/wikipedia/ru/thumb/e/e4/Barash.png/220px-Barash.png ", width=150)
prompt = st.text_input("Введите тему стиха:", "Осенний лес")

if st.button("Написать стих"):
    with st.spinner("Бараш думает..."):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        poem = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown(f"### Вот что написал Бараш:\n\n{poem}")
