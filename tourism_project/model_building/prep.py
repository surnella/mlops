# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from huggingface_hub import hf_hub_download

repo_id = "surnellas/Visit-With-Us"
filename = "tourism.csv"  # adjust to exactly match what list_repo_files printed
repo_type = "dataset"

# 1. Get token
token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN environment variable is not set")

api = HfApi(token=token)

local_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=filename,
    token=os.environ.get("HF_TOKEN"),  # needed if private
)

tourism_dataset = pd.read_csv(local_path)
print("Shape:", tourism_dataset.shape)

df = tourism_dataset.copy()

# Define the target variable for the classification task
target = 'ProdTaken'

df.drop(columns=['CustomerID',target], inplace=True)

# 1️⃣ Convert all object columns to category
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# 2️⃣ Get list of categorical columns
categorical_features = df.select_dtypes(include='category').columns.tolist()

# 3️⃣ Get list of numeric columns
numeric_features = df.select_dtypes(include=['number']).columns.tolist()

print("Categorical columns:", categorical_features)
print("Numeric columns:", numeric_features)

# Define predictor matrix (X) using selected numeric and categorical features
X = df[numeric_features + categorical_features]

# Define target variable
y =tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

repo_id = "surnellas/Visit-With-Us"
repo_type = "dataset"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type=repo_type,
    )
