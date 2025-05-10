import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Create plots directory if it doesn't exist
import os
os.makedirs('plots', exist_ok=True)

# Load the dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df['diabetes'].value_counts(normalize=True))

# Split the data
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['diabetes'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['diabetes'])

print("\nSplit sizes:")
print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(val_df)} samples")
print(f"Test set: {len(test_df)} samples")

def preprocess_data(df):
    """Preprocess the data and convert it to text format for LLM."""
    df_processed = df.copy()
    
    # Convert categorical columns to text descriptions
    gender_map = {'Male': 'male', 'Female': 'female'}
    smoking_map = {
        'never': 'never smoked',
        'current': 'currently smokes',
        'former': 'former smoker',
        'ever': 'has smoked in the past',
        'not current': 'does not currently smoke',
        'No Info': 'smoking history unknown'
    }
    
    df_processed['gender'] = df_processed['gender'].map(gender_map)
    df_processed['smoking_history'] = df_processed['smoking_history'].map(smoking_map)
    
    # Create text descriptions for each patient
    texts = []
    for _, row in df_processed.iterrows():
        text = f"Patient is {row['gender']}, {row['age']} years old, "
        text += f"with BMI of {row['bmi']:.1f}, "
        text += f"HbA1c level of {row['HbA1c_level']:.1f}, "
        text += f"blood glucose level of {row['blood_glucose_level']}, "
        text += f"and {row['smoking_history']}. "
        text += f"Heart disease status: {'has' if row['heart_disease'] else 'does not have'} heart disease. "
        text += f"Hypertension status: {'has' if row['hypertension'] else 'does not have'} hypertension."
        texts.append(text)
    
    return texts, df_processed['diabetes'].values

# Preprocess all datasets
X_train_text, y_train = preprocess_data(train_df)
X_val_text, y_val = preprocess_data(val_df)
X_test_text, y_test = preprocess_data(test_df)

# Load the BERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
print(f"\nLoading model: {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    problem_type="single_label_classification"
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

class DiabetesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
print("\nCreating datasets...")
train_dataset = DiabetesDataset(X_train_text, y_train, tokenizer)
val_dataset = DiabetesDataset(X_val_text, y_val, tokenizer)
test_dataset = DiabetesDataset(X_test_text, y_test, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
model.train()

print("\nStarting training...")
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")

# Evaluation
print("\nEvaluating model...")
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

# Calculate metrics
test_accuracy = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds)
test_roc_auc = roc_auc_score(all_labels, all_probs)

print("\nTest Set Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (BERT)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('plots/confusion_matrix_bert.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# Save the model
print("\nSaving model...")
torch.save(model.state_dict(), 'bert_diabetes_model.pt')
print("Done!") 