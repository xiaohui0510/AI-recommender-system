# run_experiment_comparison.py

import sqlite3
import time
from collections import Counter
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd

import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    top_k_accuracy_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

# ============================
# 1) HYPERPARAMETERS & SETTINGS
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5
TEMPERATURE = 1.0

# Four variants to compare:
#  • baseline:           CrossEntropy, no oversampling        (gamma=0.0, target_min=0)
#  • focal_only:         FocalLoss (γ=0.5), no oversampling   (gamma=0.5, target_min=0)
#  • oversample_only:    CrossEntropy, mild oversampling      (gamma=0.0, target_min=5)
#  • focal_oversample:   FocalLoss (γ=0.5), mild oversampling (gamma=0.5, target_min=5)
VARIANTS = {
    'baseline':         {'gamma': 0.0, 'target_min': 0},
    'focal_only':       {'gamma': 0.2, 'target_min': 0},
    'oversample_only':  {'gamma': 0.0, 'target_min': 5},
    'focal_oversample': {'gamma': 0.20, 'target_min': 5}
}

# ============================
# 2) DATA LOADING & CLEANING
# ============================
def load_and_clean_data(sqlite_path="shift_reports.db"):
    """
    Load 'shift_reports' table from SQLite, drop rows with missing 'issues' or 'resolution',
    lowercase/strip text, and return a DataFrame with ['issues', 'resolution'].
    """
    conn = sqlite3.connect(sqlite_path)
    df = pd.read_sql("SELECT * FROM shift_reports", con=conn)
    conn.close()

    df = df.dropna(how='all', axis=1)
    df_cleaned = df.dropna(how='any', axis=0)

    df_cleaned['issues'] = df_cleaned['issues'].str.lower().str.strip()
    df_cleaned['resolution'] = df_cleaned['resolution'].str.lower().str.strip()

    return df_cleaned[['issues', 'resolution']].copy()

# ============================
# 3) BERT EMBEDDING GENERATION
# ============================
def generate_bert_embeddings(texts, tokenizer, bert_model):
    """
    Given a list of input strings, return a NumPy array of shape (N, 768)
    containing [CLS] token embeddings from BERT.
    """
    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=MAX_LEN
            ).to(DEVICE)
            outputs = bert_model(**encoding)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
            embeddings.append(cls_emb.cpu().numpy())
    return np.vstack(embeddings)

# ============================
# 4) FOCAL LOSS DEFINITION
# ============================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction='none'
        )  # shape: [batch_size]
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss  # shape: [batch_size]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ============================
# 5) FNN MODEL DEFINITION
# ============================
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, extra_hidden_dim, num_classes, dropout=0.5):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, extra_hidden_dim)
        self.bn2 = nn.BatchNorm1d(extra_hidden_dim)
        self.fc3 = nn.Linear(extra_hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# ============================
# 6) TRAIN & EVALUATE FUNCTION
# ============================
def train_and_evaluate_variant(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    num_classes,
    gamma=0.0,
    target_min=0,
    return_model=False
):
    """
    Train an FNN classifier. If gamma>0, use FocalLoss; else CrossEntropy.
    If target_min>0, apply mild oversampling to bring each class up to target_min.
    If return_model=True, return (metrics_dict, model), else just metrics_dict.
    """
    # Device
    device = DEVICE

    if variant_name in ['baseline', 'focal_only']:
        train_emb_res, train_lbl_res = train_embeddings, train_labels
    else:
        original_counts = Counter(train_labels)
        sampling_strategy = {}
        if target_min > 0:
            sampling_strategy = {
                label: target_min
                for label, count in original_counts.items()
                if count < target_min
            }
        if sampling_strategy:
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            train_emb_res, train_lbl_res = ros.fit_resample(train_embeddings, train_labels)
        else:
            train_emb_res, train_lbl_res = train_embeddings, train_labels



    # ros = RandomOverSampler(random_state=42)
    # train_emb_res, train_lbl_res = ros.fit_resample(train_embeddings, train_labels)

    # 2) Class weights on (possibly oversampled) train labels
    class_weights = None if variant_name == 'baseline' else torch.tensor(
        1.0 / np.log(1.02 + np.array([Counter(train_lbl_res)[i] for i in range(num_classes)])),
        dtype=torch.float32,
        device=device
    )


    # 3) Build DataLoader
    X_train = torch.tensor(train_emb_res, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_lbl_res, dtype=torch.long).to(device)
    X_test  = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    y_test  = torch.tensor(test_labels, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 4) Model, criterion, optimizer
    model = FNN(input_dim=768, hidden_dim=256, extra_hidden_dim=256, num_classes=num_classes).to(device)
    if variant_name == 'baseline':
        criterion = nn.CrossEntropyLoss()  # no class weight
    elif gamma > 0.0:
        criterion = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5) Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Loss: {avg_loss:.4f}")

    # 6) Evaluation on test set
    model.eval()
    all_preds = []
    all_labels_list = []
    all_logits = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            scaled_logits = logits / TEMPERATURE
            probs = torch.softmax(scaled_logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_list.extend(batch_y.cpu().numpy())
            all_logits.extend(scaled_logits.cpu().numpy())

    y_true = np.array(all_labels_list)
    y_pred = np.array(all_preds)
    y_score = np.array(all_logits)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'top3_accuracy': top_k_accuracy_score(
            y_true=y_true,
            y_score=y_score,
            k=3,
            labels=np.arange(num_classes)
        )
    }

    if return_model:
        return metrics, model
    else:
        return metrics

# ============================
# 7) MAIN EXECUTION
# ============================
if __name__ == "__main__":
    print("\n🔄 Loading and cleaning data...")
    df = load_and_clean_data("shift_reports.db")

    # 7.1) Prepend "Issue:" token to each issue
    df['input_text'] = df['issues'].apply(lambda issue: f"Issue: {issue}")

    # 7.2) Encode resolution → integer label
    unique_resolutions = sorted(df['resolution'].unique())
    resolution_to_label = {res: idx for idx, res in enumerate(unique_resolutions)}
    label_to_resolution = {idx: res for res, idx in resolution_to_label.items()}
    df['label'] = df['resolution'].map(resolution_to_label)

    num_classes = len(unique_resolutions)
    print(f"Number of unique resolution classes: {num_classes}\n")

    # 7.3) Split singletons → train, stratify the rest
    counts = df['label'].value_counts()
    multi_labels = set(counts[counts >= 2].index)
    singletons_df = df[df['label'].isin(counts[counts == 1].index)]
    multi_df = df[df['label'].isin(multi_labels)]

    train_multi, test_multi = train_test_split(
        multi_df,
        test_size=0.30,
        random_state=42,
        stratify=multi_df['label']
    )

    train_df = pd.concat([train_multi, singletons_df], ignore_index=True)
    test_df = test_multi.reset_index(drop=True)

    print(f"Training samples: {len(train_df)} (including {len(singletons_df)} singletons)")
    print(f"Testing  samples: {len(test_df)}\n")

    train_texts  = train_df['input_text'].tolist()
    train_labels = train_df['label'].values
    test_texts   = test_df['input_text'].tolist()
    test_labels  = test_df['label'].values

    # 7.4) Generate BERT embeddings
    print("🔄 Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)

    print("\n⏳ Generating BERT embeddings for train set...")
    train_embeddings = generate_bert_embeddings(train_texts, tokenizer, bert_model)
    print("✅ Train embeddings shape:", train_embeddings.shape)

    print("\n⏳ Generating BERT embeddings for test set...")
    test_embeddings = generate_bert_embeddings(test_texts, tokenizer, bert_model)
    print("✅ Test embeddings shape:", test_embeddings.shape)

    # 7.5) Run each variant and save focal+oversample model
    results = {}

    for variant_name, params in VARIANTS.items():
        gamma = params['gamma']
        target_min = params['target_min']
        print(f"\n==============================")
        print(f"▶ Running variant: {variant_name}")
        print(f"   → gamma = {gamma}, target_min = {target_min}")
        print(f"==============================")

        if variant_name == 'focal_oversample':
            # return the trained model for saving
            metrics, trained_model = train_and_evaluate_variant(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                num_classes=num_classes,
                gamma=gamma,
                target_min=target_min,
                return_model=True
            )
        else:
            metrics = train_and_evaluate_variant(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                num_classes=num_classes,
                gamma=gamma,
                target_min=target_min,
                return_model=False
            )

        print(f"\n▶ Results for '{variant_name}':")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")
        results[variant_name] = metrics

        if variant_name == 'focal_oversample':
            # Save the best model and the label mapping
            torch.save(trained_model.state_dict(), "fnn_focal_oversample.pth")
            with open("label_to_resolution.pkl", "wb") as f:
                pickle.dump(label_to_resolution, f)
            print("\n✅ Saved 'fnn_focal_oversample.pth' and 'label_to_resolution.pkl'")

    # 7.6) Summary table
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df[[
        'accuracy',
        'f1_macro',
        'f1_weighted',
        'recall_macro',
        'recall_weighted',
        'top3_accuracy'
    ]]

    print("\n\n===== Experiment Comparison Summary =====")
    print(summary_df)

    # summary_df.to_csv("experiment_comparison.csv", index=True)
    # print("\n✅ All results saved to 'experiment_comparison.csv'\n")
