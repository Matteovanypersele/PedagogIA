import os
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import s3fs

# Function to download the model files from S3 using s3fs
def download_from_s3(bucket_name, s3_path, local_path):
    # Initialize s3fs with endpoint and credentials from st.secrets
    fs = s3fs.S3FileSystem(
        key=st.secrets["AWS_ACCESS_KEY_ID"],
        secret=st.secrets["AWS_SECRET_ACCESS_KEY"],
        token=st.secrets["AWS_SESSION_TOKEN"],
        client_kwargs={"endpoint_url": st.secrets["ENDPOINT_URL"]}
    )
    
    # List objects in the specified S3 path and download each one
    files = fs.ls(f"{bucket_name}/{s3_path}")
    for file in files:
        # Extract file name and download to local path
        file_name = file.split('/')[-1]
        local_file_path = os.path.join(local_path, file_name)
        fs.get(file, local_file_path)

# Function to load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    bucket_name = "matteovy"
    s3_path = "modelF5simple"  # Make sure this path is correct in S3
    local_model_path = "/tmp/modelF5simple"  # Temporary path for the model on Streamlit Cloud

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    # Download the model files from S3
    download_from_s3(bucket_name, s3_path, local_model_path)

    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate an answer from the input question
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Streamlit User Interface
st.title("Assistant Pédagogique en Mathématiques Appliquées à l'Économie")
st.write("Posez vos questions et obtenez des réponses détaillées.")

# Input field for the user’s question
question = st.text_input("Votre question :")

# Button to generate an answer
if st.button("Obtenir la réponse"):
    if question:
        with st.spinner("L'Assistant réfléchit..."):
            answer = generate_answer(question)
        st.write("Réponse : ", answer)
    else:
        st.warning("Veuillez poser une question.")
