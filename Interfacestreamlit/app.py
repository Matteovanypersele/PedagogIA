import streamlit as st
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Fonction pour charger le modèle fine-tuné et le tokenizer
@st.cache_resource
def load_model():
    model_path = "C:/Users/Tanguy/vscode/modelentraine"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Fonction pour générer une réponse à partir de la question posée
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Interface utilisateur avec Streamlit
st.title("Assistant Pédagogique en Mathématiques Appliquées à l'Économie")
st.write("Posez vos questions et obtenez des réponses détaillées.")

# Champ de saisie pour la question de l’utilisateur
question = st.text_input("Votre question :")

# Bouton pour générer la réponse
if st.button("Obtenir la réponse"):
    if question:
        with st.spinner("L'Assistant réfléchit..."):
            answer = generate_answer(question)
        st.write("Réponse : ", answer)
    else:
        st.warning("Veuillez poser une question.")
