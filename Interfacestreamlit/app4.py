import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def load_model():
    model_name = "Mvanypersele/F5basefinetunedonGSM8Ktranslatedinfrench"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

st.title("Assistant Pédagogique en Mathématiques Appliquées à l'Économie")
st.write("Posez vos questions et obtenez des réponses détaillées.")

question = st.text_input("Votre question :")

if st.button("Obtenir la réponse"):
    if question:
        with st.spinner("L'Assistant réfléchit..."):
            answer = generate_answer(question)
        st.write("Réponse : ", answer)
    else:
        st.warning("Veuillez poser une question.")
