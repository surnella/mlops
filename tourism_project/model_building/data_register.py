import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "surnellas/Visit-With-Us"
repo_type = "dataset"

# 1. Get token
token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN environment variable is not set")

api = HfApi(token=token)

# 2. Check if dataset repo exists, else create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repo...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,    # or True if you want private
    )
    print(f"Dataset '{repo_id}' created.")

# 3. Upload your local folder
print("Uploading folder tourism_project/data to the dataset repo...")
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload complete.")
