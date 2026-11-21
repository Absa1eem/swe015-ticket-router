"""Upload trained artifacts to the Hugging Face Hub so the repo stays small (R7).
Usage: huggingface-cli login ; python upload_weights.py <your-username>/ticket-router-weights"""
import sys
from huggingface_hub import HfApi
from common import ART

repo = sys.argv[1] if len(sys.argv) > 1 else "Absa1eem/ticket-router-weights"
api = HfApi()
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path=str(ART), repo_id=repo)
print("uploaded. Others can run:  HF_REPO=%s python app.py" % repo)
