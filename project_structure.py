import os

# Set the base project path
base_path = r"E:\Projects\vehicle-image-classification"

# Define the folder structure
folders = [
    "data",
    "models",
    "notebooks",
    "app",
]

# Create folders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created: {folder_path}")

# Create main files
main_files = [
    "README.md",
    "requirements.txt",
    "vehicle_classifier.py"
]

for file in main_files:
    file_path = os.path.join(base_path, file)
    with open(file_path, 'w') as f:
        f.write("")  # create an empty file
    print(f"Created file: {file_path}")

print("\n✅ Project structure created successfully!")
