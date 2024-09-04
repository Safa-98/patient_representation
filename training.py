import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_embeddings(file_path, ids, directory):
    """Load embeddings from .pt files given the list of IDs and directory."""
    return [torch.load(os.path.join(directory, f"{id_}.pt")) for id_ in ids]


def evaluate_models(X_train, Y_train, X_test, Y_test, results_file_path):
    """Train and evaluate multiple models, saving results to a file."""
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': XGBClassifier()
    }

    with open(results_file_path, "w") as results_file:
        for model_name, model in models.items():
            # Train the model
            model.fit(X_train, Y_train)

            # Make predictions on the test set
            predictions = model.predict(X_test)

            # Evaluate metrics
            accuracy = accuracy_score(Y_test, predictions)
            precision = precision_score(Y_test, predictions)
            recall = recall_score(Y_test, predictions)
            f1 = f1_score(Y_test, predictions)
            auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

            # Write metrics to the results file
            results_file.write(f"{model_name} Metrics:\n")
            results_file.write(f"Accuracy: {accuracy}\n")
            results_file.write(f"Precision: {precision}\n")
            results_file.write(f"Recall: {recall}\n")
            results_file.write(f"F1 Score: {f1}\n")
            results_file.write(f"AUC Score: {auc}\n")
            results_file.write("\n")

if __name__ == "__main__":
    # Define file paths
    train_csv = 'data/raw/train_dataset_1note.csv'
    test_csv = 'data/raw/test_dataset_1note.csv'
    pr_train_dir = 'data/processed/embed_BERT_CLS_note/tensor_BERT_CLS_avg_patient/'
    note_train_dir = 'data/processed/embed_BERT_CLS_note/'
    pr_test_dir = 'data/processed/embed_BERT_CLS_note/tensor_BERT_CLS_avg_patient/'
    note_test_dir = 'data/processed/embed_BERT_CLS_note/'
    results_file_path = "results/embed_BERT_CLS_avg_1note.txt"

    # Load train and test datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract features and labels from the datasets
    X_pr_train = load_embeddings(pr_train_dir, train_df['SUBJECT_ID'], pr_train_dir)
    X_note_train = load_embeddings(note_train_dir, train_df['NOTE_ID'], note_train_dir)
    Y_train = train_df['LABEL']

    X_pr_test = load_embeddings(pr_test_dir, test_df['SUBJECT_ID'], pr_test_dir)
    X_note_test = load_embeddings(note_test_dir, test_df['NOTE_ID'], note_test_dir)
    Y_test = test_df['LABEL']

    # Prepare feature matrices
    X_pr_train = [torch.stack(X_pr_train).numpy(), torch.stack(X_note_train).numpy()]
    X_pr_test = [torch.stack(X_pr_test).numpy(), torch.stack(X_note_test).numpy()]


    # Combine embeddings into feature matrices
    X_train = np.hstack([X_pr_train[0].reshape(X_pr_train[0].shape[0], -1), X_pr_train[1].reshape(X_pr_train[1].shape[0], -1)])
    X_test = np.hstack([X_pr_test[0].reshape(X_pr_test[0].shape[0], -1), X_pr_test[1].reshape(X_pr_test[1].shape[0], -1)])


    # Train and evaluate models
    evaluate_models(X_train, Y_train, X_test, Y_test, results_file_path)
