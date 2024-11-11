from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
import boto3
import io
import os

# Configurer la variable pour réduire la fragmentation de mémoire
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Authentification avec Hugging Face
login(token="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")


# 10. Configurer le client S3 en utilisant les variables d'environnement
s3_client = boto3.client(
    's3',
    endpoint_url="https://minio.lab.sspcloud.fr",  # Endpoint MinIO
    region_name=os.getenv('AWS_DEFAULT_REGION')
)
# Chargement du modèle et du tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token  # Définir un token de padding

# Chargement du dataset
dataset = load_dataset("gsm8k", "main", split="train")

# Prétraitement des données
def preprocess_data(examples):
    inputs = [q.strip() for q in examples['question']]
    targets = [a.strip() for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Création des ensembles d'entraînement et de validation
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': split_dataset['train'],
    'validation': split_dataset['test']
})

# Définition du collator pour le padding dynamique
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Définir les arguments d'entraînement avec les optimisations pour la mémoire
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,    # Batch réduit
    per_device_eval_batch_size=4,     # Batch réduit pour l'évaluation
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,                        # Précision mixte pour réduire la mémoire
    gradient_accumulation_steps=2,    # Accumulation de gradient pour simuler un batch plus grand
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle dans un flux mémoire pour un envoi direct vers S3
model_io = io.BytesIO()
model.save_pretrained(model_io)
model_io.seek(0)

# Sauvegarder le tokenizer dans un flux mémoire
tokenizer_io = io.BytesIO()
tokenizer.save_pretrained(tokenizer_io)
tokenizer_io.seek(0)

# Uploader le modèle et le tokenizer dans S3
bucket_name = 'matteovy'
s3_model_path = 'my_model/'

s3_client.upload_fileobj(model_io, bucket_name, f'{s3_model_path}model.bin')
s3_client.upload_fileobj(tokenizer_io, bucket_name, f'{s3_model_path}tokenizer')
print("Modèle et tokenizer sauvegardés directement sur S3 !")
