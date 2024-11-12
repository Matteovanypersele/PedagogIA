import os
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import boto3
from botocore.client import Config  # Import Config pour éviter l'erreur

# Fonction pour télécharger le modèle depuis S3
def download_from_s3(bucket_name, s3_path, local_path):
    # Initialiser le client S3 avec l'endpoint personnalisé et les identifiants
    s3 = boto3.client(
        "s3",
        endpoint_url=st.secrets["ENDPOINT_URL"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
        config=Config(signature_version="s3v4", region_name="us-east-1")  # Signature v4 pour MinIO
    )

    # Lister et télécharger les fichiers
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_path)
    for obj in objects.get('Contents', []):
        file_key = obj['Key']
        file_name = file_key.split('/')[-1]
        s3.download_file(bucket_name, file_key, os.path.join(local_path, file_name))

# Fonction pour charger le modèle fine-tuné et le tokenizer
@st.cache_resource
def load_model():
    bucket_name = "matteovy"
    s3_path = "modelF5simple"  # Vérifiez que c'est bien le bon chemin dans S3
    local_model_path = "/tmp/modelF5simple"  # Emplacement temporaire pour le modèle

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


