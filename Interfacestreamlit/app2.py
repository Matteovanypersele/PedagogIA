import os
import subprocess
import sys

# Fonction pour installer les packages nécessaires
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installation des bibliothèques si nécessaires
try:
    import streamlit as st
except ImportError:
    install("streamlit")

try:
    import transformers
except ImportError:
    install("transformers")
try:
    import datasets
except ImportError:
    install("datasets")
try:
    import streamlit as st
except ImportError:
    install("transformers[torch]")

try:
    import boto3
except ImportError:
    install("boto3")

import streamlit as st
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import boto3

# Crée une fonction pour télécharger le modèle depuis S3
def download_from_s3(bucket_name, s3_path, local_path):
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_path)
    for obj in objects.get('Contents', []):
        file_key = obj['Key']
        file_name = file_key.split('/')[-1]
        s3.download_file(bucket_name, file_key, os.path.join(local_path, file_name))

# Fonction pour charger le modèle fine-tuné et le tokenizer
@st.cache_resource
def load_model():
    bucket_name = "matteovy"
    s3_path = "modelF5simple"
    local_model_path = "/tmp/modelF5simple"  # Emplacement temporaire pour le modèle sur Streamlit Cloud

    # Créer le dossier local si nécessaire
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    # Télécharger le modèle depuis S3
    download_from_s3(bucket_name, s3_path, local_model_path)

    # Charger le modèle et le tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
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