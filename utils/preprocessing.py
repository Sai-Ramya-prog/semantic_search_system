import os
import re

DATA_PATH = "data/20_newsgroups"

def clean_text(text):
    lines = text.split("\n")
    
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start = i + 1  
    
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            next_content = ""
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip():
                    next_content = lines[j].strip()
                    break
            
            if next_content and not re.match(r"^[\w\-]+:\s", next_content):
                body_start = i + 1
                break
            else:
                body_start = i + 1
    
    text = " ".join(lines[body_start:])
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def load_dataset():

    documents = []
    labels = []
    label_names = []

    for label_id, category in enumerate(os.listdir(DATA_PATH)):

        category_path = os.path.join(DATA_PATH, category)

        if os.path.isdir(category_path):
            print(f"\nLoading category {label_id}: {category}")

            label_names.append(category)

            for file in os.listdir(category_path):

                file_path = os.path.join(category_path, file)

                try:
                    with open(file_path, "r", encoding="latin1") as f:

                        text = f.read()
                        text = clean_text(text)

                        documents.append(text)
                        labels.append(label_id)

                except:
                    continue

    return documents, labels, label_names