from huggingface_hub import HfApi

api = HfApi()

# Tady vyplň své údaje
repo_id = "vojmahdal/roberta-sentiment-3labels" 

print("Nahrávám model na Hugging Face... Může to trvat několik minut.")

api.upload_folder(
    folder_path="./sentiment_analysis_model",  # Nahrát vše v aktuální složce
    repo_id=repo_id,
    repo_type="model",
    # Ignorujeme zbytečnosti, co nechceme na internetu
    ignore_patterns=["checkpoint-*", ".venv/*", "upload.py", ".git*"]
)

print(f"Hotovo! Model najdeš na: https://huggingface.co/{repo_id}")