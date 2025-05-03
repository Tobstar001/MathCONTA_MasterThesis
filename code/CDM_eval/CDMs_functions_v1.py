# CDMs_functions_v1.py

"""
Custom Functions Library

This file contains user-defined functions organized into 7 categories:
1. General
2. CV (Cross Validation)
3. minK
4. CDD (Class Distribution Divergence)
5. n_gram_acc (N-Gram Accuracy)
6. conta_traces (Contamination Traces)
7. Visualization
"""
import random
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
import json
from datetime import datetime
import pytz
import itertools
from itertools import cycle, product
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from pathlib import Path
import statistics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest


# ===========================
# 0. CONSTANT
# ===========================
DRIVE_PATH=Path('/content/drive/MyDrive/Masterarbeit25/')

# ===========================
# 1. General
# ===========================
def list_files_in_directory(base_path, model_id):
    """
    Prints all file names in the directory with only the last two components of their path.

    Parameters:
        base_path (str or Path): Directory to list files from.

    Returns:
        List[str]: List of file names.
    """
    base_path = Path(base_path)/ model_id

    if not base_path.exists() or not base_path.is_dir():
        print(f"{base_path} is not a valid directory.")
        return

    file_paths=[]
    print(f"Files in {base_path}:\n")
    for file in base_path.rglob('*'):
        if file.is_file():
            parts = file.parts[-2:]  # last two directories + filename
            print(Path(*parts))
            file_paths.append(file)

    return file_paths


def truncate_ids(data, verbose=False):
    """
    Processes the data structure by truncating each inner list in the 'SAMPLE_IDS' key of each dictionary
    such that any id after (and including) the first occurrence of 100257 is removed.

    Optionally prints the mean length of the inner lists before and after truncation when verbose is True.

    Parameters:
        data (list): List of dictionaries, where each dictionary contains a 'SAMPLE_IDS' key
                     mapping to a list of inner lists of ids.
        verbose (bool): Whether to print information about mean list lengths (default: False).

    Returns:
        list: The modified data structure with truncated id lists.
    """
    # Print mean lengths before truncation if verbose.
    if verbose:
        print("Before Truncation:")
    for idx, data_dict in enumerate(data):
        sample_ids = data_dict.get('SAMPLE_IDS', [])
        # Compute the mean length of inner lists.
        if sample_ids:
            mean_length_before = statistics.mean(len(inner_list) for inner_list in sample_ids)
        else:
            mean_length_before = 0
        if verbose:
            print(f"Dictionary {idx}: Mean length = {mean_length_before}")

    # Process each dictionary and truncate each inner list after the first occurrence of 100257.
    for idx, data_dict in enumerate(data):
        new_sample_ids = []
        for id_list in data_dict.get('SAMPLE_IDS', []):
            if 100257 in id_list:
                # Find the index of the first occurrence of 100257.
                idx_val = id_list.index(100257)
                # Truncate list up to and including 100257.
                new_sample_ids.append(id_list[:idx_val + 1])
            else:
                new_sample_ids.append(id_list)
        data_dict['SAMPLE_IDS'] = new_sample_ids

    # Print mean lengths after truncation if verbose.
    if verbose:
        print("\nAfter Truncation:")
    for idx, data_dict in enumerate(data):
        sample_ids = data_dict.get('SAMPLE_IDS', [])
        if sample_ids:
            mean_length_after = statistics.mean(len(inner_list) for inner_list in sample_ids)
        else:
            mean_length_after = 0
        if verbose:
            print(f"Dictionary {idx}: Mean length = {mean_length_after}")

    return data
def truncate_ids_ngram_cdd(ngram_cdd_data, verbose=False):
  for dict_out in ngram_cdd_data:
      # Access the dictionary under "NGRAM_CDD_GENERATION_DATA"
      for key, dict_out2 in dict_out["NGRAM_CDD_GENERATION_DATA"].items():
          # This will allow iterating over all keys, including the "10" key
          for dict_out3 in dict_out2["cdd_generation_data"]:
              dict_out3=truncate_ids([dict_out3],verbose=verbose)[0]

def merge_json_files(file_paths, output_file):
    """
    Merges multiple JSON files containing lists of dictionaries into a single JSON file.

    Args:
        file_paths (list): A list of paths to the JSON files to merge.
        output_file (str): The path to the output JSON file.
    """
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)  # Extend the list with data from each file

    with open(output_file, 'w') as f:
        json.dump(merged_data, f)

def save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict):
    dir = Path(log_path_base) / model_id / method_name
    dir.mkdir(parents=True, exist_ok=True)

    file_path = dir / f"{data_name}_accuracylog_{exp_id}.json"

    with open(file_path, "a") as acc_file:
        json.dump(out_dict, acc_file, indent=4)

    print(f"Accuracy log saved in {file_path}")

#-minK + Conta_Traces-#
def create_mathconta_token_data(model_id, ds_conta=None, model=None, tokenizer=None, only_problem=False,
                                force_reprocess=False, data_name="MathCONTA",base_path_token=DRIVE_PATH / "MathCONTA_tokens"):
    """
    Processes a Hugging Face Dataset by generating token-level information for each row
    using a given language model and tokenizer. The results are cached to disk and reloaded
    if already available, unless forced to reprocess.

    Parameters:
        ds_conta (datasets.Dataset): Hugging Face dataset with columns including
            'PROBLEM', 'SOLUTION', 'ID', 'CATEGORY', 'LABEL', and 'LABEL_BINARY' or None if not used.
        model (PreTrainedModel): Hugging Face-compatible language model used for inference or None if not used.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model or None if not used
        model_id (str): Identifier used to create a subdirectory for storing cached token data.
        only_problem (bool): If True, use only the 'PROBLEM' field as input prompt.
            If False, use 'PROBLEM' + 'SOLUTION'.
        force_reprocess (bool): If True, reprocess the data even if cached file exists.
        base_path_token (Path or str): Base directory for saving/loading the cached token data w/o model_id
        for example and Default: '/content/drive/MyDrive/Masterarbeit25/MathCONTA_tokens'

    Returns:
        List[dict]: A list of dictionaries containing token-level information for each row,
            including input ids, decoded tokens, token probabilities, and log-likelihoods.
    """
    base_path = Path(base_path_token)
    dir = base_path / model_id
    file_path = dir / f'{data_name}_token_data.json'

    if file_path.exists() and not force_reprocess:
        print(f"Loading existing token data from: {file_path}")
        with open(file_path, 'r') as f:
            MathCONTA_token_data = json.load(f)
        return MathCONTA_token_data

    print("Processing token data from scratch...")
    MathCONTA_token_data = []

    for row in ds_conta:
        prompt = row["PROBLEM"] if only_problem else row["PROBLEM"] + row["SOLUTION"]

        input_ids, logits = get_inference_logits(prompt, model, tokenizer)
        token_id_list = input_ids[0].tolist()

        decoded_tokens, token_probs = get_token_probabilities(tokenizer, logits, input_ids)
        _, token_log_likes = get_token_loglikelihoods(tokenizer, logits, input_ids)

        entry = {
            'ID': row['ID'],
            'CATEGORY': row['CATEGORY'],
            'PROBLEM': row['PROBLEM'],
            'LABEL': row['LABEL'],
            'SOLUTION': row['SOLUTION'],
            'LABEL_BINARY': row['LABEL_BINARY'],
            'input_ids_list': token_id_list,
            'decoded_tokens': decoded_tokens,
            'token_probs': token_probs,
            'token_log_likes': token_log_likes
        }

        MathCONTA_token_data.append(entry)

    dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(MathCONTA_token_data, f)

    print(f"Processed token data saved to: {file_path}")
    return MathCONTA_token_data

def stratified_dict_split(data, test_ratio=0.2, seed=42):
    # Step 1: Filter to only valid entries
    valid_entries = [
        d for d in data
        if all(k in d for k in ['ID', 'CATEGORY', 'LABEL_BINARY'])
    ]

    if not valid_entries:
        raise ValueError("No entries with all of 'ID', 'CATEGORY', 'LABEL_BINARY' found.")

    # Step 2: Create DataFrame with required fields
    df = pd.DataFrame([
        {'ID': d['ID'], 'CATEGORY': d['CATEGORY'], 'LABEL_BINARY': d['LABEL_BINARY']}
        for d in valid_entries
    ])

    # Step 3: Sort by ID for reproducibility
    df['ID'] = df['ID'].astype(str)
    df = df.sort_values('ID').reset_index(drop=True)

    # Step 4: Create stratification key
    df['stratify_key'] = df['CATEGORY'].astype(str) + "_" + df['LABEL_BINARY'].astype(str)

    # Step 5: Stratified split
    train_ids, test_ids = train_test_split(
        df['ID'],
        test_size=test_ratio,
        stratify=df['stratify_key'],
        random_state=seed
    )

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    # Step 6: Split original data using ID
    train_data = [d for d in data if str(d.get('ID')) in train_id_set]
    test_data  = [d for d in data if str(d.get('ID')) in test_id_set]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    print(test_id_set)
    return train_data, test_data

def filter_random_examples(dataset, num_examples=5, seed=None):
    """Filters a dataset to select a specified number of random examples.

    Args:
      dataset: The dataset to filter.
      num_examples: The number of random examples to select.
      seed: An optional seed for the random number generator (for reproducibility).

    Returns:
      A filtered dataset containing the selected random examples.
    """
    if seed is not None:
        random.seed(seed)  # Set the seed if provided

    unique_ids = dataset.unique("ID")
    random_ids = random.sample(list(unique_ids), num_examples)
    filtered_dataset = dataset.filter(lambda example: example["ID"] in random_ids)

    return filtered_dataset

#-eval on testset-#
def evaluate_method_standard(cdd_df, metric_col, theta, model_id, data_name, method_name, parameter1=None,
                             parameter2=None, n_bootstrap=1000, seed=42):
    """
    Evaluates a model's binary classification performance against a majority-class baseline,
    with metrics and significance testing.
    """
    import numpy as np
    from datetime import datetime
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from scipy.stats import binomtest

    np.random.seed(seed)

    # Extract ground truth and binarized predictions
    y_true = cdd_df['LABEL_BINARY'].astype(int).values

    # Exponential branch: parameter1 is 'exponential', theta is a tuple (theta_a, theta_b),
    # and metric_col is a tuple of two column names.
    if "ensemble" in method_name.lower():
        y_pred = cdd_df[metric_col].astype(int).values
    elif parameter1 == "exponential":
        if not (isinstance(metric_col, tuple) and len(metric_col) == 2):
            raise ValueError("For exponential fit, metric_col must be a tuple of two column names.")
        if not (isinstance(theta, tuple) and len(theta) == 2):
            raise ValueError("For exponential fit, theta must be a tuple of two thresholds (theta_a, theta_b).")
        theta_a, theta_b = theta
        # Decision rule: contaminated if A_value < theta_a OR B_value > theta_b.
        y_pred = ((cdd_df[metric_col[0]] < theta_a) | (cdd_df[metric_col[1]] > theta_b)).astype(int).values
    else:
        # Default branch: metric_col is a single column and theta is a numeric threshold.
        y_pred = (cdd_df[metric_col] > theta).astype(int).values

    # Baseline: constant majority class prediction
    majority_class = 1
    y_base = np.full_like(y_true, fill_value=majority_class)

    # Accuracy of model predictions
    acc = accuracy_score(y_true, y_pred)

    # Bootstrapped confidence interval for accuracy
    boot_accs = [
        accuracy_score(y_true[indices], y_pred[indices])
        for indices in [np.random.choice(len(y_true), size=len(y_true), replace=True) for _ in range(n_bootstrap)]
    ]
    acc_ci_lower = np.percentile(boot_accs, 2.5)
    acc_ci_upper = np.percentile(boot_accs, 97.5)

    # McNemar’s test to assess significance vs. baseline
    b = np.sum((y_pred == y_true) & (y_base != y_true))  # Model correct, baseline wrong
    c = np.sum((y_pred != y_true) & (y_base == y_true))  # Model wrong, baseline correct
    mcnemar_p = binomtest(k=min(b, c), n=b + c, p=0.5, alternative='two-sided').pvalue if (b + c) > 0 else 1.0

    # Precision, recall, F1, and confusion matrix
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Return full evaluation record
    return {
        "model_id": model_id,
        "data_name": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter": {
            "parameter1": parameter1,
            "parameter2": parameter2,
            "theta": str(theta)
        },
        "metrics": {
            'accuracy': float(acc),
            'accuracy_95CI': (float(acc_ci_lower), float(acc_ci_upper)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.astype(int).tolist(),
            'mcnemar_b': int(b),
            'mcnemar_c': int(c),
            'mcnemar_p_value': float(mcnemar_p),
        },
        "n_bootstrap": n_bootstrap,
        "seed": seed
    }

# ===========================
# 2. CV (Cross Validation)
# ===========================
#-CV-1D-#
def compute_candidate_thresholds(x_train, y_train):
    """
    Compute candidate thresholds from the training set.
    Thresholds are computed as midpoints between consecutive sorted training samples
    where the label changes.

    If no threshold is found (e.g., all labels are identical), thresholds outside the range are returned.
    Here we use an offset of 0.1 from the minimum and maximum values.

    Parameters:
        x_train (np.array): Training feature values.
        y_train (np.array): Corresponding training labels.

    Returns:
        candidate_thresholds (list): List of candidate thresholds.
    """
    sorted_indices = np.argsort(x_train)
    x_sorted = x_train[sorted_indices]
    y_sorted = y_train[sorted_indices]

    candidate_thresholds = []
    for i in range(len(x_sorted) - 1):
        if y_sorted[i] != y_sorted[i + 1]:
            candidate_thresholds.append((x_sorted[i] + x_sorted[i + 1]) / 2)

    # If no candidate thresholds found, use thresholds outside the training range with an offset of 0.1.
    if not candidate_thresholds:
        candidate_thresholds = [x_sorted[0] - 0.1, x_sorted[-1] + 0.1]

    return candidate_thresholds

#-CV-#
def evaluate_threshold_on_dataset(threshold, x, y):
    """
    Evaluate a given threshold on a dataset (x, y) using the decision rule:
      - Predict 0 (uncontaminated) if x < threshold, else predict 1 (contaminated).

    Parameters:
        threshold (float): The threshold to evaluate.
        x (np.array): Feature values.
        y (np.array): True labels.

    Returns:
        accuracy (float): Classification accuracy.
    """
    predictions = (x >= threshold).astype(int)
    accuracy = np.mean(predictions == y)
    return accuracy

#-CV-#
def select_best_threshold(x_train, y_train, candidate_thresholds):
    """
    From candidate thresholds, select the threshold that yields the highest training set accuracy.

    Parameters:
        x_train (np.array): Training feature values.
        y_train (np.array): Training labels.
        candidate_thresholds (list): List of candidate thresholds.

    Returns:
        best_thresh (float): The candidate threshold with the highest training accuracy.
        best_train_acc (float): The corresponding training accuracy.
    """
    best_thresh = None
    best_train_acc = -1

    for thresh in candidate_thresholds:
        acc_train = evaluate_threshold_on_dataset(thresh, x_train, y_train)
        if acc_train > best_train_acc:
            best_train_acc = acc_train
            best_thresh = thresh

    return best_thresh, best_train_acc

#-CV-1D#
def find_optimal_threshold_from_df_cv(df, feature_col, label_col="label", cv_folds=5, random_state=42, verbose=False):
    """
    Performs stratified k-fold cross-validation to find the optimal threshold for binary classification
    using a single numeric feature.

    Returns:
        optimal_thresholds (list): Optimal thresholds for each fold.
        train_accuracies (list): Training accuracies per fold.
        test_accuracies (list): Test accuracies per fold.
        global_optimal_threshold (float): Optimal threshold computed from the full dataset.
    """
    x = df[feature_col].to_numpy()
    y = df[label_col].to_numpy()

    optimal_thresholds = []
    train_accuracies = []
    test_accuracies = []

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold_num = 1
    for train_index, test_index in skf.split(x.reshape(-1, 1), y):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        candidate_thresholds = compute_candidate_thresholds(x_train, y_train)
        best_thresh, best_train_acc = select_best_threshold(x_train, y_train, candidate_thresholds)
        test_acc = evaluate_threshold_on_dataset(best_thresh, x_test, y_test)

        optimal_thresholds.append(best_thresh)
        train_accuracies.append(best_train_acc)
        test_accuracies.append(test_acc)

        if verbose:
            print(f"Fold {fold_num}: Optimal threshold = {best_thresh}, "
                  f"Training accuracy = {best_train_acc:.4f}, Test accuracy = {test_acc:.4f}")
        fold_num += 1

    # Compute global optimal threshold using the full dataset
    global_candidates = compute_candidate_thresholds(x, y)
    global_optimal_threshold, _ = select_best_threshold(x, y, global_candidates)

    return optimal_thresholds, train_accuracies, test_accuracies, global_optimal_threshold

#-CV-2D-#
def compute_candidate_thresholds_2d(a_train, b_train, y_train):
    """
    Compute candidate thresholds independently for features 'a' and 'b'.

    Parameters:
        a_train (np.array): Training values for feature 'a'.
        b_train (np.array): Training values for feature 'b'.
        y_train (np.array): Binary training labels.

    Returns:
        tuple: Two lists containing candidate thresholds for 'a' and 'b'.
    """
    a_thresholds = compute_candidate_thresholds(a_train, y_train)
    b_thresholds = compute_candidate_thresholds(b_train, y_train)
    return a_thresholds, b_thresholds

#-CV-2D-#
def evaluate_thresholds_on_dataset_2d(a_delta, b_delta, a_vals, b_vals, labels):
    """
    Evaluate classification accuracy for the rule:
        Predict 1 if a < a_delta OR b > b_delta, else 0.

    Parameters:
        a_delta (float): Threshold for feature 'a'.
        b_delta (float): Threshold for feature 'b'.
        a_vals (np.array): Feature 'a' values.
        b_vals (np.array): Feature 'b' values.
        labels (np.array): Ground truth binary labels.

    Returns:
        float: Accuracy score.
    """
    predictions = ((a_vals < a_delta) | (b_vals > b_delta)).astype(int)
    return np.mean(predictions == labels)

#-CV-2D-#
def select_best_thresholds_2d(a_train, b_train, y_train, a_thresholds, b_thresholds):
    """
    Find the (a_delta, b_delta) pair that yields the highest training accuracy.

    Parameters:
        a_train (np.array): Training values for feature 'a'.
        b_train (np.array): Training values for feature 'b'.
        y_train (np.array): Training binary labels.
        a_thresholds (list): Candidate thresholds for 'a'.
        b_thresholds (list): Candidate thresholds for 'b'.

    Returns:
        tuple: ((a_delta, b_delta), best_accuracy)
    """
    best_acc = -1
    best_pair = (None, None)

    for a_delta, b_delta in product(a_thresholds, b_thresholds):
        acc = evaluate_thresholds_on_dataset_2d(a_delta, b_delta, a_train, b_train, y_train)
        if acc > best_acc:
            best_acc = acc
            best_pair = (a_delta, b_delta)

    return best_pair, best_acc

#-CV-2D-#
def find_optimal_2d_threshold_cv(df, a_col, b_col, label_col="label", cv_folds=5, random_state=42, verbose=False):
    """
    Perform stratified k-fold cross-validation to find optimal (a_delta, b_delta)
    thresholds for binary classification using a rule-based model:
        Predict 1 if a < a_delta OR b > b_delta, else 0.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and labels.
        a_col (str): Column name for feature 'a'.
        b_col (str): Column name for feature 'b'.
        label_col (str): Column name for binary label.
        cv_folds (int): Number of cross-validation folds.
        random_state (int): Seed for reproducibility.
        verbose (bool): If True, print fold details.

    Returns:
        tuple: (list of (a_delta, b_delta) per fold, list of train accuracies, list of test accuracies)
    """
    x_a = df[a_col].to_numpy()
    x_b = df[b_col].to_numpy()
    y = df[label_col].to_numpy()

    best_thresholds = []
    train_accuracies = []
    test_accuracies = []

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    fold = 1
    for train_idx, test_idx in skf.split(df[[a_col, b_col]], y):
        a_train, b_train, y_train = x_a[train_idx], x_b[train_idx], y[train_idx]
        a_test, b_test, y_test = x_a[test_idx], x_b[test_idx], y[test_idx]

        a_thresholds, b_thresholds = compute_candidate_thresholds_2d(a_train, b_train, y_train)
        (a_delta, b_delta), train_acc = select_best_thresholds_2d(a_train, b_train, y_train, a_thresholds, b_thresholds)
        test_acc = evaluate_thresholds_on_dataset_2d(a_delta, b_delta, a_test, b_test, y_test)

        best_thresholds.append((a_delta, b_delta))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if verbose:
            print(f"Fold {fold}: a_delta = {a_delta}, b_delta = {b_delta}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
        fold += 1

    # Compute global optimal threshold using the full dataset
    global_candidates_a, global_candidates_b = compute_candidate_thresholds_2d(x_a, x_b, y)
    (global_optimal_threshold_a, global_optimal_threshold_b), _ = select_best_thresholds_2d(x_a, x_b, y, global_candidates_a, global_candidates_b)

    return best_thresholds, train_accuracies, test_accuracies, (global_optimal_threshold_a, global_optimal_threshold_b)

# ===========================
# 3. minK
# ===========================
#-minK-#
def get_minK_df_from_tokendata(list_data, k=20):
    """
    Returns a pd.DataFrame with columns: ID, CATEGORY, LABEL, LABEL_BINARY, k, minK.

    Args:
        list_data (list): A list of dictionaries (stored on Drive!!)
        k (int, optional): Value for minK calculation. Defaults to 20.

    Returns:
        pd.DataFrame: DataFrame with minK values.
    """
    data = []
    for list_element in list_data:
        token_probs = list_element['token_probs']
        minK = get_mink_value(token_probs, k)
        data.append({
            'ID': list_element['ID'],
            'CATEGORY': list_element['CATEGORY'],
            'LABEL': list_element['LABEL'],
            'LABEL_BINARY': list_element['LABEL_BINARY'],
            'k': k,  # Store the k value
            'minK_value': minK
        })
    return pd.DataFrame(data)

#-minK-#
def get_mink_value(token_probs, k=20):
    """
    This function takes a list of probabilities and returns the sum of the k% smallest probabilities.
    Source: [1] W. Shi et al., “Detecting Pretraining Data from Large Language Models,”

    Args:
        token_probs (list): A list of probability values.
        k (float): A percentage (0 to 100) representing the proportion of the smallest values to sum.

    Returns:
        float: The sum of the k% smallest probabilities.
    """
    # Ensure k is between 0 and 100
    if not (0 <= k <= 100):
        raise ValueError("k must be a percentage between 0 and 100.")

    # Sort the probabilities in ascending order
    sorted_probs = sorted(token_probs)

    # Determine the number of elements corresponding to k%
    num_elements = int(len(sorted_probs) * (k / 100))

    # Sum the k% smallest probabilities
    mink_value_sum = sum(sorted_probs[:num_elements])

    # Normalize the value
    mink_value = mink_value_sum/num_elements

    return mink_value

#-minK-#
def tune_minK(tokendata, feature_col, label_col, k_range, cv_folds, model_id, data_name, seed, method_name,
              log_path_base=Path(DRIVE_PATH / "not_specified"), exp_id='exp_id'):
    """
    Evaluates the best parameter set for a given dataset to optimize cross-validated accuracy.

    This function iteratively evaluates different values of `k` by generating a filtered dataset,
    computing the optimal decision threshold using cross-validation, and selecting the best `k`
    based on the highest mean test accuracy across CV folds.

    Args:
        tokendata (pd.DataFrame): Token-level input data.
        feature_col (str): Name of the column used as feature input.
        label_col (str): Name of the column containing the ground truth labels.
        k_range (Iterable[int]): Range of values for the minimum token frequency threshold to be evaluated.
        cv_folds (int): Number of cross-validation folds.
        model_id (str): Identifier for the model or experiment group.
        data_name (str): Name of the dataset being used.
        seed (int): Random seed for reproducibility in cross-validation.
        method_name (str): Name of the method or algorithm being tested.
        log_path_base (Path, optional): Base path for saving logs w/o model_id: e.g: DRIVE_PATH / "cdm_data" / "MathCONTA_v1"
                                        logs will be found: log_path_base / model_id / method_name
        exp_id (str, optional): Identifier for the experiment. Defaults to 'exp_id'.

    Returns:
        dict: A dictionary containing:
            - model_id (str)
            - data_set (str)
            - datetime (str): ISO format timestamp
            - method_name (str)
            - parameter_range (dict): Range of `k` and indication that threshold is optimized
            - CV_folds (int)
            - CV_seed (int)
            - best_log_entry (dict): Log entry for the best-performing `k`
            - all_log_entries (list): Log entries for all tested `k` values

    Side Effects:
        - Prints progress and best result to the console.
        - Saves accuracy logs as a JSON file to the specified `DRIVE_PATH`.

    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for k in k_range:
        # Get the DataFrame for current k
        minK_df = get_minK_df_from_tokendata(tokendata, k)

        # Get the optimal threshold and CV metrics
        optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
            minK_df, feature_col, label_col, cv_folds
        )

        # Format values: round to 6 decimal places and cast to native float
        median_threshold_fmt = round(float(statistics.median(optimal_thresholds)), 6)
        mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
        mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)
        theta_glob_fmt = round(float(theta_glob), 6)

        # Print progress
        print(f"Testing k={k} | median_threshold={median_threshold_fmt:.6f} | mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

        # Log result
        log_entry = {
            "method": method_name,
            "parameter": {"k": k},
            "global_threshold": theta_glob_fmt,
            "median_threshold": median_threshold_fmt,
            "mean_cvacc_train": mean_cvacc_train_fmt,
            "mean_cvacc_test": mean_cvacc_test_fmt,
            "all_thresholds": [round(float(th), 6) for th in optimal_thresholds],
            "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
            "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
        }
        results.append(log_entry)

        # Update best result
        if mean_cvacc_test_fmt > best_score:
            best_score = mean_cvacc_test_fmt
            best_log_entry = log_entry

    # Print best log entry (main info only)
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "global_threshold": best_log_entry["global_threshold"],
        "median_threshold": best_log_entry["median_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'k':str(k_range),'theta':"optimized"},
        "CV_folds": cv_folds,
        "CV_seed": seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }
    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)
    return out_dict

#-minK-#
def get_inference_logits(prompt, model, tokenizer):
    """
    This function takes a prompt and returns the logits for each token in the prompt.

    input:
    prompt: str
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer

    output:
    input_ids: torch.Tensor (torch.Size([1, len(tokens)]))
    logits: torch.Tensor (torch.Size([1, len(tokens), vocab_size])
    """
    tokens = tokenizer(prompt, return_tensors="pt")

    input_ids = tokens["input_ids"] #torch.Size([1, 10])

    # Move input tokens (=tensors) to GPU
    tokens = {key: value.to("cuda") for key, value in tokens.items()}

    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**tokens)
    logits = outputs.logits #torch.Size([1, 10, 128256])

    return input_ids, logits

#-minK-#
def get_token_probabilities(tokenizer, logits, input_ids, debug=False):
    """
    This function takes logits and input_ids and returns the probabilities for each token.

    input:
    logits: torch.Tensor (torch.Size([1, len(tokens), vocab_size])
    input_ids: torch.Tensor (torch.Size([1, len(tokens)]))
    debug: bool

    output:
    decoded_tokens: list
    token_probs: list
    """
    # Shift logits and input IDs for token-level probabilities
    shifted_logits = logits[:, :-1, :]  # Exclude last token's prediction: The last (n_th) prediction is measured on the n-1 logits!
    shifted_input_ids = input_ids[:, 1:]  # Exclude first token (<|begin_of_text|>) as a label

    # having all on cpu:
    shifted_logits = shifted_logits.to("cpu")

    # Compute probabilities
    probabilities = torch.nn.functional.softmax(shifted_logits, dim=-1) #torch.Size([1, 9, 128256]), dim=-1: along last dimension


    # Gather probabilities for actual tokens
    token_probs = probabilities.gather(dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    # index tensor should have same number of dimensions as probabilities tensor -> unsqueeze.
    # after that we squeeze it back so that token_probs have torch.Size([1,9])

    # Decode tokens and print them with probabilities
    decoded_tokens = tokenizer.convert_ids_to_tokens(shifted_input_ids.squeeze().tolist())
    token_probs = token_probs.squeeze().tolist()

    if debug:
        for i, (token, prob) in enumerate(zip(decoded_tokens, token_probs)):
            print(f"Token: {token}, Probability: {prob:.4f}")

    return decoded_tokens, token_probs

#-minK-#
def get_token_loglikelihoods(tokenizer, logits, input_ids, debug=False):
    """
    This function takes logits and input_ids and returns the log_likelihood for each token. (alternative approach!!)

    input:
    logits: torch.Tensor (torch.Size([1, len(tokens), vocab_size])
    input_ids: torch.Tensor (torch.Size([1, len(tokens)]))
    debug: bool

    output:
    decoded_tokens: list
    token_loglikes: list
    """
    # Shift logits and input IDs for token-level probabilities
    shifted_logits = logits[:, :-1, :]  # Exclude last token's prediction: The last (n_th) prediction is measured on the n-1 logits!
    shifted_input_ids = input_ids[:, 1:]  # Exclude first token (<|begin_of_text|>) as a label
    # having all on cpu:
    shifted_logits = shifted_logits.to("cpu")

    # Compute probabilities
    token_logs = torch.nn.functional.log_softmax(shifted_logits, dim=-1) #torch.Size([1, 9, 128256]), dim=-1: along last dimension


    # Gather probabilities for actual tokens
    token_logs = token_logs.gather(dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    # index tensor should have same number of dimensions as probabilities tensor -> unsqueeze.
    # after that we squeeze it back so that token_logs have torch.Size([1,9])

    # Decode tokens and print them with probabilities
    decoded_tokens = tokenizer.convert_ids_to_tokens(shifted_input_ids.squeeze().tolist())
    token_logs = token_logs.squeeze().tolist()

    if debug:
        for i, (token, prob) in enumerate(zip(decoded_tokens, token_logs)):
            print(f"Token: {token}, LogLike: {prob:.4f}")

    return decoded_tokens, token_logs

#-minK-#
def get_most_likely_token_probabilities(tokenizer, logits,debug=False):
    """
    This function takes logits and returns the most likely token and its probability.

    input:
    logits: torch.Tensor (torch.Size([1, len(tokens), vocab_size])
    debug: bool

    output:
    most_likely_tokens: list
    most_likely_probs: list
    """

    # Shift logits and input IDs for token-level probabilities
    shifted_logits = logits[:, :-1, :]  # Exclude last token's prediction: The last (n_th) prediction is measured on the n-1 logits!

    # having all on cpu:
    shifted_logits = shifted_logits.to("cpu")

    # Compute probabilities
    probabilities = torch.nn.functional.softmax(shifted_logits, dim=-1) #torch.Size([1, 9, 128256]), dim=-1: along last dimension

    # Determine the most likely token for each step
    most_likely_ids = torch.argmax(probabilities, dim=-1)  # Get the indices of the most likely tokens
    most_likely_tokens = tokenizer.convert_ids_to_tokens(most_likely_ids.squeeze().tolist())
    most_likely_probs = torch.max(probabilities, dim=-1).values.squeeze().tolist()

    # Print tokens with their probabilities and most likely tokens
    if debug:
        for i, (most_likely, most_prob) in enumerate(zip(most_likely_tokens, most_likely_probs)):
            print(f"  Most Likely Token: {most_likely}, Probability: {most_prob:.4f}")

    return most_likely_tokens, most_likely_probs



#-minK-#
def run_minK_on_prompt(input_minK, model, tokenizer, k, theta, verbose=False):
    """

    :param input_minK: text which incorporates Q + Agold
    :param model:
    :param tokenizer:
    :param k: value between 0 and 100 (how many tokens percentage-wise should be in the minK set)
    :param theta: if minK is higher than theta -> it is flaged as contaminated
    :param verbose: Print the tokens with their probability
    :return: output, minK value

    """
    input_ids, logits = get_inference_logits(input_minK, model, tokenizer)
    decoded_tokens, token_probs = get_token_probabilities(tokenizer, logits, input_ids, verbose)
    minK = get_mink_value(token_probs, k)

    output = minK > theta

    return output, minK

#-minK-#
def run_minK_on_minK_df(minK_df, theta):
    """
    Compares minK_value to threshold theta and evaluates prediction performance.

    Args:
        minK_df (pd.DataFrame): DataFrame containing 'minK_value' and 'LABEL_BINARY' columns.
        theta (float): Threshold for binarizing the minK_value.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, confusion_matrix.
    """
    # Predict: 1 if minK_value > theta, else 0
    predictions = (minK_df['minK_value'] > theta).astype(int)

    # Ground truth
    ground_truth = minK_df['LABEL_BINARY'].astype(int)

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
    }

#-minK-#
def run_minK(model_id, ds_conta, model, tokenizer,
             dir_token_path, k=5, theta=0.0002,
             only_problem=False, force_reprocess=False):
    """
    Full pipeline for minK-based classification evaluation on a dataset.

    This function:
    1. Loads or generates token-level data using `create_mathconta_token_data`.
    2. Computes minK values for each example using `get_minK_df_from_tokendata`.
    3. Compares minK values against a threshold (theta) using `run_minK_on_minK_df`.
    4. Returns evaluation metrics including accuracy, precision, recall, and F1-score.

    Args:
        model_id (str): Identifier for the model used, also used to name the token cache folder.
        ds_conta (datasets.Dataset): Hugging Face dataset containing required columns:
            'PROBLEM', 'SOLUTION', 'ID', 'CATEGORY', 'LABEL', and 'LABEL_BINARY'.
        model (PreTrainedModel): Hugging Face-compatible language model used for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        dir_token_path (str or Path): Directory path for storing or loading cached token data.
        k (int, optional): The k-value used in minK calculation. Defaults to 5.
        theta (float, optional): Threshold applied to minK values for binary classification. Defaults to 0.0002.
        only_problem (bool, optional): Whether to use only the problem text as model input. Defaults to False.
        force_reprocess (bool, optional): If True, regenerate token data even if cache exists. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Overall classification accuracy
            - 'precision': Precision of the positive class
            - 'recall': Recall of the positive class
            - 'f1_score': F1-score of the positive class
    """

    # Step 1: Generate or load token-level data
    MathCONTA_token_data = create_mathconta_token_data(
        model_id=model_id,
        ds_conta=ds_conta,
        model=model,
        tokenizer=tokenizer,
        only_problem=only_problem,
        force_reprocess=force_reprocess,
        base_path_token=dir_token_path
    )

    # Step 2: Compute minK values into a DataFrame
    minK_df = get_minK_df_from_tokendata(MathCONTA_token_data, k=k)

    # Step 3: Evaluate with provided theta
    all_metrics = run_minK_on_minK_df(minK_df, theta=theta)

    # Step 4: Return selected metrics
    return {
        'accuracy': all_metrics['accuracy'],
        'precision': all_metrics['precision'],
        'recall': all_metrics['recall'],
        'f1_score': all_metrics['f1_score']
    }





# ===========================
# 4. CDD
# ===========================
#-Cdd generation-#
def create_cdd_generation_data(model_id, ds_conta=None, model=None, tokenizer=None,
                               sample_size=3, max_new_tokens=30, torch_seed=42,
                               force_reprocess=False, verbose=False,data_name="MathCONTA",
                               base_path_token=DRIVE_PATH / "MathCONTA_cdd_generation_data"):

    """
    Processes a Hugging Face Dataset by generating cdd generation data for each row
    using the given language model and tokenizer. The results are cached to disk and
    reloaded if already available, unless forced to reprocess.

    Parameters:
        ds_conta (datasets.Dataset): Hugging Face dataset with columns including
            'PROBLEM', 'SOLUTION', 'ID', 'CATEGORY', 'LABEL', and 'LABEL_BINARY'.
        model (PreTrainedModel): The language model used for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        model_id (str): Identifier used to create a subdirectory for storing cached cdd generation data.
        sample_size (int): Number of sample outputs to generate per prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        torch_seed (int): Seed for the torch random number generator.
        force_reprocess (bool): If True, reprocess the data even if a cached file exists.
        verbose (bool): If True, prints additional output during generation.
        base_path_token (Path or str): Base directory for saving/loading the cached cdd generation data.

    Returns:
        List[dict]: A list of dictionaries containing cdd generation data for each row,
                    including original dataset fields and generation outputs:
                    - "greedy_ids": Token IDs for the greedy output.
                    - "sample_ids": List of token ID lists for the sample outputs.
                    - "max_length": The maximum token length among outputs.
    """
    base_path = Path(base_path_token)
    cache_dir = base_path / model_id
    file_path = cache_dir / f"{data_name}_sample{sample_size}_max{max_new_tokens}_seed{torch_seed}.json"

    if file_path.exists() and not force_reprocess:
        print(f"Loading existing cdd generation data from: {file_path}")
        with open(file_path, 'r') as f:
            cdd_data = json.load(f)
        return cdd_data

    print("Processing cdd generation data from scratch...")
    cdd_data = []

    # Set random seed for reproducibility
    torch.manual_seed(torch_seed)

    for row in ds_conta:
        # Construct the prompt by only using 'PROBLEM'
        print(f"Processing ID:{row['ID']}")
        prompt = row["PROBLEM"]
        # Generate cdd generation data using the helper function.
        generation_data = generate_cdd_data_row(prompt, model, tokenizer, sample_size,
                                                max_new_tokens=max_new_tokens, verbose=verbose)
        # Combine the generation output with original dataset fields.
        entry = {
            'ID': row['ID'],
            'CATEGORY': row['CATEGORY'],
            'PROBLEM': row['PROBLEM'],
            'SOLUTION': row['SOLUTION'],
            'LABEL': row['LABEL'],
            'LABEL_BINARY': row["LABEL_BINARY"]}
        entry.update(generation_data)
        cdd_data.append(entry)

    # Ensure the cache directory exists and then save the data.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(cdd_data, f)

    print(f"Processed cdd generation data saved to: {file_path}")
    return cdd_data


def generate_cdd_data_row(prompt, model, tokenizer, sample_size, max_new_tokens=30, verbose=False, batch_size=25):
    """
    Generates outputs using the given model and tokenizer, returning a cdd_generation_data object.

    Args:
        prompt (str): The input prompt for the language model.
        model: The pre-trained language model.
        tokenizer: The tokenizer associated with the language model.
        sample_size (int): The number of sample outputs to generate.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 30.
        verbose (bool, optional): If True, prints the decoded outputs. Defaults to False.
        batch_size (int, optional): If provided and >= 2, batches the generation of sample outputs.

    Returns:
        dict: A cdd_generation_data object with keys:
            - "GREEDY_IDS": List of token IDs for the greedy output.
            - "SAMPLE_IDS": List of token ID lists for the sample outputs.
            - "MAX_LENGTH": The maximum token length among outputs.
    """
    # Tokenize and move inputs to the appropriate device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    # Generate greedy output (non-sampling)
    greedy_output = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    # Extract newly generated tokens (token IDs) from the greedy output
    generated_tokens = greedy_output[:, inputs["input_ids"].shape[-1]:]
    greedy_ids = generated_tokens[0].tolist()

    if verbose:
        greedy_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print("Greedy Output:")
        print(greedy_text)

    sample_ids = []  # List to hold each sample's token ID list
    max_length = len(greedy_ids)

    # If batch_size is not provided or less than 2, fall back to the simple loop
    if batch_size is None or batch_size < 2:
        for i in range(sample_size):
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
            generated_tokens = output[:, inputs["input_ids"].shape[-1]:]
            current_ids = generated_tokens[0].tolist()
            sample_ids.append(current_ids)

            if verbose:
                output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                print(f"Sample {i + 1} Output:")
                print(output_text)

            if len(current_ids) > max_length:
                max_length = len(current_ids)
    else:
        # Generate samples in batches.
        remaining = sample_size
        sample_counter = 0
        while remaining > 0:
            print("batch")
            current_batch_size = batch_size if remaining >= batch_size else remaining
            # Use num_return_sequences to generate a batch of sample outputs
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=current_batch_size
            )
            # The generated output is now batched. Unbatch each sample.
            # generated_tokens is of shape (current_batch_size, sequence_length)
            for j in range(current_batch_size):
                # Slice off the prompt tokens
                generated_tokens_j = output[j, inputs["input_ids"].shape[-1]:]
                current_ids = generated_tokens_j.tolist()
                sample_ids.append(current_ids)
                sample_counter += 1

                if verbose:
                    output_text = tokenizer.decode(generated_tokens_j, skip_special_tokens=True)
                    print(f"Sample {sample_counter} Output:")
                    print(output_text)

                if len(current_ids) > max_length:
                    max_length = len(current_ids)
            remaining -= current_batch_size

    cdd_generation_data_row = {
        "MAX_LENGTH": max_length,
        "GREEDY_IDS": greedy_ids,
        "SAMPLE_IDS": sample_ids
    }
    return cdd_generation_data_row


#-Cdd calculation-#
def token_edit_distance(a, b):
    """
    Calculate the token-level edit distance between two token lists a and b.

    Parameters:
        a (list): List of tokens (numbers).
        b (list): List of tokens (numbers).

    Returns:
        int: The edit distance between the two token lists.
    """
    len_a, len_b = len(a), len(b)

    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    curr = [0] * (len_b + 1)

    for i in range(1, len_a + 1):
        curr[0] = i
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[len_b]

# Function to calculate rho_star
def calculate_rho_star(greedy_output, output_list):
    """
    Calculates ρ*(d), the proportion of outputs with edit distance d from the greedy output.

    Args:
        greedy_output (list): The greedy output token sequence (list of ints).
        output_list (list): A list of generated output token sequences (lists of ints).

    Returns:
        dict: A dictionary where keys are edit distances (d) and values are ρ*(d).
    """
    output_edit_list = [token_edit_distance(greedy_output, output) for output in output_list]

    rho_star = {}
    for d in set(output_edit_list):
        rho_star[d] = output_edit_list.count(d) / len(output_list)

    return rho_star

# Function to calculate peak cumulative diversity density (CDD)
def calculate_peak_cdd(rho_star, max_length, alpha=0.05):
    """
    Calculates the peak cumulative diversity density (CDD) efficiently.

    Args:
        rho_star (dict): A dictionary where keys are edit distances (d) and values are ρ*(d).
        max_length (int): The maximum token length among outputs.
        alpha (float, optional): The threshold for inclusion in the peak CDD calculation.

    Returns:
        float: The peak cumulative diversity density.
    """
    threshold = alpha * max_length
    peak_cdd = sum(value for key, value in rho_star.items() if key <= threshold)
    return peak_cdd

def get_cdd_df_from_generation_data(cdd_generation_data, alpha=0.05):
    """
    Generate a DataFrame of computed CDD (Conditional Distance Discrepancy) values
    from generation data based on token-level edit distance statistics.

    For each dictionary in the input list, the function extracts the 'GREEDY_IDS',
    'SAMPLE_IDS', and 'MAX_LENGTH' values, computes token-level edit distance statistics
    using `calculate_rho_star`, and then determines the peak CDD value using
    `calculate_peak_cdd` with the provided significance level (alpha). The results
    along with relevant metadata are compiled into a pandas DataFrame.

    Parameters
    ----------
    cdd_generation_data : list of dict
        A list where each element is a dictionary containing generation data.
        Each dictionary should include the following keys:
          - 'ID': Unique identifier for the data entry.
          - 'CATEGORY': Category of the data.
          - 'LABEL': Descriptive label.
          - 'LABEL_BINARY': Binary label indicator.
          - 'GREEDY_IDS': List of token IDs from a greedy generation process.
          - 'SAMPLE_IDS': List of token IDs from a sampling-based generation process.
          - 'MAX_LENGTH': Maximum sequence length considered.
    alpha : float, optional
        Significance level used in computing the peak CDD value (default is 0.05).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
          - 'ID': Identifier from the input data.
          - 'CATEGORY': Category from the input data.
          - 'LABEL': Label from the input data.
          - 'LABEL_BINARY': Binary label indicator from the input data.
          - 'alpha': The significance level used in the computation.
          - 'cdd_value': The computed peak CDD value for the corresponding data entry.
    """
    data = []
    for list_element in cdd_generation_data:
        greedy_ids = list_element["GREEDY_IDS"]
        sample_ids = list_element["SAMPLE_IDS"]
        max_length = list_element["MAX_LENGTH"]

        # Compute the token-level edit distance statistics
        rho_star_dict = calculate_rho_star(greedy_ids, sample_ids)
        peak_cdd = calculate_peak_cdd(rho_star_dict, max_length, alpha)

        data.append({
            'ID': list_element['ID'],
            'CATEGORY': list_element['CATEGORY'],
            'LABEL': list_element['LABEL'],
            'LABEL_BINARY': list_element['LABEL_BINARY'],
            'alpha': alpha,
            'cdd_value': peak_cdd
        })
    return pd.DataFrame(data)

def tune_cdd(cdd_generation_data, feature_col, label_col, alpha_range, cv_folds, cv_seed, model_id, data_name, method_name,
              log_path_base=Path(DRIVE_PATH / "not_specified"), exp_id='exp_id'):
    """
    Evaluates the best parameter set for a given dataset to optimize cross-validated accuracy.

    This function iteratively evaluates different values of `alpha`.
    Computing the optimal decision threshold using cross-validation, and selecting the best `alpha`
    (and the median threshold) based on the highest mean test accuracy across CV folds.

    Args:
        cdd_generation_data (pd.DataFrame): cdd_generation_data (see create_cdd_generation_data())
        feature_col (str): Name of the column used as feature input.
        label_col (str): Name of the column containing the ground truth labels.
        alpha_range (Iterable[int]): Range of values for the minimum token frequency threshold to be evaluated.
        cv_folds (int): Number of cross-validation folds.
        model_id (str): Identifier for the model or experiment group.
        data_name (str): Name of the dataset being used.
        seed (int): Random seed for reproducibility in cross-validation.
        method_name (str): Name of the method or algorithm being tested.
        log_path_base (Path, optional): Base path for saving logs w/o model_id: e.g: DRIVE_PATH / "cdm_data" / "MathCONTA_v1"
                                        logs will be found: log_path_base / model_id / method_name
        exp_id (str, optional): Identifier for the experiment. Defaults to 'exp_id'.

    Returns:
        dict: A dictionary containing:
            - model_id (str)
            - data_set (str)
            - datetime (str): ISO format timestamp
            - method_name (str)
            - parameter_range (dict): Range of `k` and indication that threshold is optimized
            - CV_folds (int)
            - CV_seed (int)
            - best_log_entry (dict): Log entry for the best-performing `k`
            - all_log_entries (list): Log entries for all tested `k` values

    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for alpha in alpha_range:
        # Get the DataFrame for current k
        cdd_df = get_cdd_df_from_generation_data(cdd_generation_data, alpha)

        # Get the optimal threshold and CV metrics
        optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
            cdd_df, feature_col, label_col, cv_folds
        )

        # Format values: round to 6 decimal places and cast to native float
        median_threshold_fmt = round(float(statistics.median(optimal_thresholds)), 6)
        mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
        mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)
        theta_glob_fmt = round(float(theta_glob), 6)

        # Print progress
        print(f"Testing alpha={alpha} | median_threshold={median_threshold_fmt:.6f} | mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

        # Log result
        log_entry = {
            "method": method_name,
            "parameter": {"alpha": alpha},
            "global_threshold": theta_glob_fmt,
            "median_threshold": median_threshold_fmt,
            "mean_cvacc_train": mean_cvacc_train_fmt,
            "mean_cvacc_test": mean_cvacc_test_fmt,
            "all_thresholds": [round(float(th), 6) for th in optimal_thresholds],
            "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
            "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
        }
        results.append(log_entry)

        # Update best result
        if mean_cvacc_test_fmt > best_score:
            best_score = mean_cvacc_test_fmt
            best_log_entry = log_entry

    # Print best log entry (main info only)
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "global_threshold": best_log_entry["global_threshold"],
        "median_threshold": best_log_entry["median_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'alpha':str(alpha_range),'theta':"optimized"},
        "CV_folds": cv_folds,
        "CV_seed": cv_seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }
    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)
    return out_dict

def run_CDD(model_id, ds_conta, model, tokenizer,
             dir_token_path, data_name="MathCONTA", alpha=5, theta=0.0002,
            force_reprocess=False,max_new_tokens=10,sample_size=3):
    """
    Full pipeline for minK-based classification evaluation on a dataset.

    This function:
    1. Loads or generates token-level data using `create_mathconta_token_data`.
    2. Computes minK values for each example using `get_minK_df_from_tokendata`.
    3. Compares minK values against a threshold (theta) using `run_minK_on_minK_df`.
    4. Returns evaluation metrics including accuracy, precision, recall, and F1-score.

    Args:
        model_id (str): Identifier for the model used, also used to name the token cache folder.
        ds_conta (datasets.Dataset): Hugging Face dataset containing required columns:
            'PROBLEM', 'SOLUTION', 'ID', 'CATEGORY', 'LABEL', and 'LABEL_BINARY'.
        model (PreTrainedModel): Hugging Face-compatible language model used for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        dir_token_path (str or Path): Directory path for storing or loading cached token data.
        k (int, optional): The k-value used in minK calculation. Defaults to 5.
        theta (float, optional): Threshold applied to minK values for binary classification. Defaults to 0.0002.
        only_problem (bool, optional): Whether to use only the problem text as model input. Defaults to False.
        force_reprocess (bool, optional): If True, regenerate token data even if cache exists. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Overall classification accuracy
            - 'precision': Precision of the positive class
            - 'recall': Recall of the positive class
            - 'f1_score': F1-score of the positive class
    """

    # Step 1: Generate or load token-level data
    data = create_cdd_generation_data(
        model_id=model_id,
        ds_conta=ds_conta,
        model=model,
        tokenizer=tokenizer,
        force_reprocess=force_reprocess,
        base_path_token=dir_token_path,
        max_new_tokens=max_new_tokens,sample_size=sample_size, data_name=data_name
    )

    # Step 2: Compute minK values into a DataFrame
    cdd_df = get_cdd_df_from_generation_data(data, alpha=alpha)

    # Step 3: Evaluate with provided theta
    all_metrics = run_cdd_on_cdd_df(cdd_df, theta=theta)

    # Step 4: Return selected metrics
    return {
        'accuracy': all_metrics['accuracy'],
        'precision': all_metrics['precision'],
        'recall': all_metrics['recall'],
        'f1_score': all_metrics['f1_score']
    }

def run_cdd_on_cdd_df(cdd_df, theta):
    """
    Compares cdd_value to threshold theta and evaluates prediction performance.

    Args:
        cdd_df (pd.DataFrame): DataFrame containing 'cdd_value' and 'LABEL_BINARY' columns.
        theta (float): Threshold for binarizing the cdd_value.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, confusion_matrix.
    """
    # Predict: 1 if minK_value > theta, else 0
    predictions = (cdd_df['cdd_value'] > theta).astype(int)

    # Ground truth
    ground_truth = cdd_df['LABEL_BINARY'].astype(int)

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
    }


# ===========================
# 5a. n_gram_acc (N-Gram Accuracy)
# ===========================
def select_start_points(tokens, n_start, n, start_offset, random_starts, verbose):
    """
    Determines and returns a list of starting token positions based on the parameters,
    performing all necessary checks and adjustments.

    Args:
        tokens: Tensor of token IDs.
        n_start (int): The number of starting points to generate.
        n (int): The size of the n-gram.
        start_offset (int): Minimum number of tokens to skip at the start.
        random_starts (bool): If True, randomly select starting points; otherwise, distribute evenly.
        verbose (bool): If True, prints warnings and debug information.

    Returns:
        list: A list of starting token positions.

    Raises:
        ValueError: If the token sequence is too short for the given n and start_offset.
    """
    verbose = True
    token_length = tokens.size(0)

    # Adjust start_offset if input is too short
    if token_length <= start_offset + n:
        adjusted_offset = max(1, min(start_offset, token_length // 3))
        if verbose:
            print(f"Warning: Input text too short. Adjusted start_offset from {start_offset} to {adjusted_offset}.")
        start_offset = adjusted_offset

    available_range = token_length - start_offset - n
    if available_range <= 0:
        raise ValueError(
            f"Token sequence (length {token_length}) is too short for the given N-gram size ({n}) and offset ({start_offset}).")

    candidate_points = list(range(start_offset, token_length - n))

    if len(candidate_points) < n_start:
        if verbose:
            print(
                f"Warning: Only {len(candidate_points)} possible start points available, but n_start={n_start}. Using all available points.")
        n_start = len(candidate_points)

    if random_starts:
        return random.sample(candidate_points, n_start)
    else:
        if n_start == 1:
            return [start_offset + available_range // 2]
        else:
            indices = np.linspace(0, len(candidate_points) - 1, num=n_start, dtype=int)
            return [candidate_points[i] for i in indices]


def generate_ngram_for_start(start, tokens, inputs, model, n, temp, tokenizer, verbose):
    """
    Generates the target and predicted n-grams for a single start point.

    Args:
        start (int): The starting token position.
        tokens: Tensor of token IDs.
        inputs: Tokenizer output containing input_ids and possibly attention_mask.
        model: Pre-trained language model.
        n (int): The size of the n-gram.
        temp (float): The temperature parameter for model sampling.
        tokenizer: The tokenizer for decoding.
        verbose (bool): If True, prints generation details.

    Returns:
        tuple: (target_ngram, predicted_ngram)
    """
    current_inputs = copy.deepcopy(inputs)
    current_inputs["input_ids"] = tokens[:start].unsqueeze(0)
    if "attention_mask" in current_inputs:
        current_inputs["attention_mask"] = current_inputs["attention_mask"][:, :start]

    target_ngram = tokens[start:start + n].tolist()

    with torch.no_grad():
        outputs = model.generate(
            **current_inputs,
            max_new_tokens=n,
            do_sample=True if temp > 0.0 else False,
            temperature=temp if temp > 0.0 else None,
            pad_token_id=tokenizer.eos_token_id
        )

    predicted_ngram = outputs[0, -n:].tolist()

    if verbose:
        prompt_text = tokenizer.decode(tokens[:start], skip_special_tokens=True)
        predicted_text = tokenizer.decode(predicted_ngram, skip_special_tokens=True)
        actual_text = tokenizer.decode(target_ngram, skip_special_tokens=True)
        print(f"Start point {start}:\nPrompt: {prompt_text}\nPredicted: {predicted_text}\nActual: {actual_text}\n")

    return target_ngram, predicted_ngram


def generate_ngram_data_row(input_text, model, tokenizer, n_start, n, temp,
                            random_starts=True, start_offset=5, seed_rand=42,
                            seed_torch=42, verbose=False):
    """
    Generates target and predicted n-grams using the given model and tokenizer.

    Args:
        input_text (str): The input text sequence.
        model: The pre-trained language model.
        tokenizer: The tokenizer associated with the language model.
        n_start (int): The number of starting points to generate.
        n (int): The size of the n-gram (e.g., 3 for trigrams).
        temp (float): The temperature parameter for model sampling.
        random_starts (bool): If True, randomly select starting points. If False, distribute evenly.
        start_offset (int): Minimum number of tokens to skip at the start.
        seed_rand (int): Random seed for Python's random module.
        seed_torch (int): Random seed for torch.
        verbose (bool): If True, prints generation details.

    Returns:
        dict: A dictionary with keys:
            - "TARGET_NGRAMS": List of token ID sequences from the actual text.
            - "PREDICTED_NGRAMS": List of token ID sequences generated by the model.
            - "START_POINTS": List of starting token positions used.
    """
    # Set seeds for reproducibility
    random.seed(seed_rand)
    torch.manual_seed(seed_torch)

    # Tokenize the input sequence
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        model = model.to("cuda")

    tokens = inputs["input_ids"].squeeze()

    # Determine candidate start points using the helper function (with all checks)
    start_points = select_start_points(tokens, n_start, n, start_offset, random_starts, verbose)

    target_ngrams = []
    predicted_ngrams = []

    # Generate n-grams for each starting point
    for start in start_points:
        target_ngram, predicted_ngram = generate_ngram_for_start(start, tokens, inputs, model, n, temp, tokenizer,
                                                                 verbose)
        target_ngrams.append(target_ngram)
        predicted_ngrams.append(predicted_ngram)

    return {
        "TARGET_NGRAMS": target_ngrams,
        "PREDICTED_NGRAMS": predicted_ngrams,
        "START_POINTS": start_points
    }


def create_ngram_generation_data(model_id, ds_conta=None, model=None, tokenizer=None,
                                 n_starts=3, n=[30], temp=1.0, torch_seed=42, random_starts=True,
                                 seed_rand=42, start_offset=5, only_problem=False,
                                 force_reprocess=False, verbose=False,
                                 data_name="MathCONTA", base_path_token=DRIVE_PATH / "MathCONTA_ngram_generation_data"):
    """
    Processes a Hugging Face Dataset by generating n-gram generation data for each row
    using the given language model and tokenizer. The n-gram size parameter 'n' is now a list,
    and generation is performed for each n value. The results are cached to disk and
    reloaded if already available, unless forced to reprocess.

    Parameters:
        ds_conta (datasets.Dataset): Hugging Face dataset with columns including
            'PROBLEM', 'SOLUTION', 'ID', 'CATEGORY', 'LABEL', and optionally 'LABEL_BINARY'.
        model (PreTrainedModel): The language model used for generation.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        model_id (str): Identifier used to create a subdirectory for storing cached n-gram generation data.
        n_starts (int): Number of starting points to generate per prompt.
        n (list): List of n-gram sizes to generate.
        temp (float): Temperature parameter for model sampling.
        torch_seed (int): Seed for the torch random number generator.
        seed_rand (int): Seed for the Python random module.
        start_offset (int): Minimum number of tokens to skip at the start.
        force_reprocess (bool): If True, reprocess the data even if a cached file exists.
        verbose (bool): If True, prints additional output during generation.
        data_name (str): Name used in the cache file.
        base_path_token (Path or str): Base directory for saving/loading the cached n-gram generation data.

    Returns:
        List[dict]: A list of dictionaries containing n-gram generation data for each row,
                    including original dataset fields and generation outputs:
                    - "ngram_generation_data": A dictionary where each key is an n value (as string)
                      and each value is the generation data with keys:
                        - "target_ngrams": List of token ID sequences from the actual text.
                        - "predicted_ngrams": List of token ID sequences generated by the model.
                        - "start_points": List of starting token positions used.
    """
    base_path = Path(base_path_token)
    cache_dir = base_path / model_id
    n_str = "_".join(map(str, n))
    file_path = cache_dir / f"{data_name}_nstarts{n_starts}_n{n_str}_seed{torch_seed}.json"
    print(f"File path: {file_path}")
    if file_path.exists() and not force_reprocess:
        print(f"Loading existing n-gram generation data from: {file_path}")
        with open(file_path, 'r') as f:
            ngram_data = json.load(f)
        for entry in ngram_data:
            entry["NGRAM_GENERATION_DATA"] = {int(k): v for k, v in entry["NGRAM_GENERATION_DATA"].items()}
        return ngram_data

    print("Processing n-gram generation data from scratch...")
    ngram_data = []

    # Set torch seed for reproducibility
    torch.manual_seed(torch_seed)

    # Process each row in the dataset
    for row in ds_conta:
        print(f"Processing ID: {row['ID']}")

        prompt = row["PROBLEM"] if only_problem else row["PROBLEM"] + row["SOLUTION"]
        generation_results = {}

        # Iterate over each n-gram size in the list
        for n_val in n:
            generation_data = generate_ngram_data_row(
                input_text=prompt,
                model=model,
                tokenizer=tokenizer,
                n_start=n_starts,
                n=n_val,
                temp=temp,
                random_starts=random_starts,
                start_offset=start_offset,
                seed_rand=seed_rand,
                seed_torch=torch_seed,
                verbose=verbose
            )
            generation_results[n_val] = generation_data

        # Combine the generation outputs with original dataset fields
        entry = {
            'ID': row['ID'],
            'CATEGORY': row['CATEGORY'],
            'PROBLEM': row['PROBLEM'],
            'SOLUTION': row['SOLUTION'],
            'LABEL': row['LABEL'],
            'LABEL_BINARY': row.get("LABEL_BINARY", None)
        }
        entry.update({'NGRAM_GENERATION_DATA': generation_results})
        ngram_data.append(entry)

    # Ensure the cache directory exists and then save the data.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ngram_data, f)

    print(f"Processed n-gram generation data saved to: {file_path}")
    return ngram_data


def calculate_ngram_acc(target_ngrams, predicted_ngrams):
    """
    Calculates the n-gram accuracy by computing the fraction of target n-grams
    that match the predicted n-grams at the same position.

    Parameters:
        target_ngrams (list): The list of target n-grams (ground truth).
        predicted_ngrams (list): The list of predicted n-grams.

    Returns:
        float: The accuracy as a fraction between 0 and 1.
    """
    correct = 0
    total = 0

    for target, predicted in zip(target_ngrams, predicted_ngrams):
        if target == predicted:
            correct += 1
        total += 1

    if total == 0:
        return 0.0

    accuracy = correct / total
    return accuracy


def get_ngram_df_from_generation_data(ngram_generation_data, n):
    """
    Constructs a pandas DataFrame from a list of generation data dictionaries,
    computing the n-gram accuracy for each entry.

    Parameters:
        ngram_generation_data (list): List of dictionaries containing n-gram generation data.
        n (int): The n-gram size.

    Returns:
        pd.DataFrame: A DataFrame with columns including ID, CATEGORY, LABEL, LABEL_BINARY, n, and ngram_acc.

    Raises:
        ValueError: If the specified n is not found in the NGRAM_GENERATION_DATA of any entry.
    """
    data = []

    for entry in ngram_generation_data:
        ngram_data_dict = entry.get("NGRAM_GENERATION_DATA", {})

        if n not in ngram_data_dict:
            available_ns = list(ngram_data_dict.keys())
            raise ValueError(
                f"Requested n={n} not found in NGRAM_GENERATION_DATA for ID {entry.get('ID', 'UNKNOWN')}. "
                f"Available n values: {available_ns}"
            )

        ngram_data = ngram_data_dict[n]
        target_ngrams = ngram_data["TARGET_NGRAMS"]
        predicted_ngrams = ngram_data["PREDICTED_NGRAMS"]

        ngram_acc = calculate_ngram_acc(target_ngrams, predicted_ngrams)

        data.append({
            'ID': entry['ID'],
            'CATEGORY': entry['CATEGORY'],
            'LABEL': entry['LABEL'],
            'LABEL_BINARY': entry['LABEL_BINARY'],
            'n': n,
            'ngram_acc': ngram_acc
        })

    return pd.DataFrame(data)

def tune_ngram_acc(ngram_data, feature_col, label_col, n_range, cv_folds, model_id, data_name, cv_seed, method_name,
              log_path_base=Path(DRIVE_PATH / "not_specified"), exp_id='exp_id'):
    """

    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for n in n_range:
        # Get the DataFrame for current k
        ngram_df = get_ngram_df_from_generation_data(ngram_data, n)

        # Get the optimal threshold and CV metrics
        optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
            ngram_df, feature_col, label_col, cv_folds, random_state=cv_seed
        )

        # Format values: round to 6 decimal places and cast to native float
        median_threshold_fmt = round(float(statistics.median(optimal_thresholds)), 6)
        mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
        mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)
        theta_glob_fmt = round(float(theta_glob), 6)

        # Print progress
        print(f"Testing n={n} | median_threshold={median_threshold_fmt:.6f} | mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

        # Log result
        log_entry = {
            "method": method_name,
            "parameter": {"n": n},
            "global_threshold": theta_glob_fmt,
            "median_threshold": median_threshold_fmt,
            "mean_cvacc_train": mean_cvacc_train_fmt,
            "mean_cvacc_test": mean_cvacc_test_fmt,
            "all_thresholds": [round(float(th), 6) for th in optimal_thresholds],
            "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
            "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
        }
        results.append(log_entry)

        # Update best result
        if mean_cvacc_test_fmt > best_score:
            best_score = mean_cvacc_test_fmt
            best_log_entry = log_entry

    # Print best log entry (main info only)
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "global_threshold": best_log_entry["global_threshold"],
        "median_threshold": best_log_entry["median_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'n':str(n_range),'theta':"optimized"},
        "CV_folds": cv_folds,
        "CV_seed": cv_seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }
    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)
    return out_dict


def run_ngram(model_id, filtered_ds, model, tokenizer, data_name,
                           force_reprocess=False, n_starts=1, n_range=[1], temp=1.0,
                           torch_seed=42, seed_rand=42, start_offset=5, random_starts=True,
                           only_problem=False, verbose=False, dir_token_path=None, n=1, theta=0.5):
    """
    Generates token-level data, computes n-gram statistics, evaluates metrics, and returns results.

    Parameters:
        model_id (str): Identifier for the model.
        filtered_ds: Filtered dataset for generation.
        model: Language model to use.
        tokenizer: Tokenizer corresponding to the model.
        data_name (str): Name for the generated data.
        force_reprocess (bool): Whether to regenerate data even if cached.
        n_starts (int): Number of starting points for generation.
        n_range (tuple): Range of n-gram lengths to consider.
        temp (float): Sampling temperature.
        torch_seed (int): Torch seed for reproducibility.
        seed_rand (int): Seed for random module.
        start_offset (int): Offset for sequence start.
        random_starts (bool): If True, randomize starting positions.
        only_problem: Optionally focus on a specific problem.
        verbose (bool): Verbosity flag.
        dir_token_path (str): Path for storing or loading token-level data.
        n (int): Specific n-gram size for evaluation.
        theta (float): Threshold parameter for evaluation metrics.

    Returns:
        dict: Evaluated metrics based on n-gram statistics.
    """
    # Step 1: Generate or load token-level data
    data = create_ngram_generation_data(
        model_id=model_id,
        ds_conta=filtered_ds,
        model=model,
        tokenizer=tokenizer,
        data_name=data_name,
        force_reprocess=force_reprocess,
        n_starts=n_starts,
        n=n_range,
        temp=temp,
        torch_seed=torch_seed,
        seed_rand=seed_rand,
        start_offset=start_offset,
        random_starts=random_starts,
        only_problem=only_problem,
        verbose=verbose,
        base_path_token=dir_token_path
    )

    # Step 2: Compute minK values into a DataFrame
    ngram_df = get_ngram_df_from_generation_data(data, n=n)

    # Step 3: Evaluate with provided theta
    all_metrics = run_on_ngram_df(ngram_df, theta=theta)

    # Step 4: Return selected metrics
    return all_metrics

def run_on_ngram_df(df, theta):
    """
    Compares cdd_value to threshold theta and evaluates prediction performance.

    Args:
        cdd_df (pd.DataFrame): DataFrame containing 'cdd_value' and 'LABEL_BINARY' columns.
        theta (float): Threshold for binarizing the cdd_value.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, confusion_matrix.
    """
    # Predict: 1 if minK_value > theta, else 0
    predictions = (df['ngram_acc'] > theta).astype(int)

    # Ground truth
    ground_truth = df['LABEL_BINARY'].astype(int)

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
    }

# ===========================
# 5b. n_gram_loglike
# ===========================
def get_ngram_loglike_df_from_generation_data(ngram_generation_data, n):
    """
    """
    data = []

    for entry in ngram_generation_data:
        ngram_data_dict = entry.get("NGRAM_LOGLIKE_GENERATION_DATA", {})

        if n not in ngram_data_dict:
            available_ns = list(ngram_data_dict.keys())
            raise ValueError(
                f"Requested n={n} not found in NGRAM_LOGLIKE_GENERATION_DATA for ID {entry.get('ID', 'UNKNOWN')}. "
                f"Available n values: {available_ns}"
            )

        ngram_data = ngram_data_dict[n]
        ngram_loglikes = ngram_data["NGRAM_LOGLIKES"]

        ngram_loglike = np.mean(ngram_loglikes)

        data.append({
            'ID': entry['ID'],
            'CATEGORY': entry['CATEGORY'],
            'LABEL': entry['LABEL'],
            'LABEL_BINARY': entry['LABEL_BINARY'],
            'n': n,
            'ngram_loglike': ngram_loglike
        })

    return pd.DataFrame(data)


def generate_ngram_loglike_data_row(input_loglikes, n_start, n, random_starts=True,
                                    start_offset=5, seed_rand=42, verbose=False, prompt_decoded=None):
    """
    """

    # Determine candidate start points using the helper function (with all checks)
    start_points = select_start_points(torch.Tensor(input_loglikes), n_start, n, start_offset, random_starts, verbose)

    ngram_loglikes = []
    decoded_ngrams = []

    # Generate n-grams for each starting point
    for start in start_points:
        sum_loglikes = sum(input_loglikes[start:start + n])
        ngram_loglikes.append(sum_loglikes)
        if prompt_decoded is not None:
            decoded_ngrams.append(prompt_decoded[start:start + n])

    return {
        "NGRAM_LOGLIKES": ngram_loglikes,
        "DECODED_NGRAMS": decoded_ngrams if prompt_decoded is not None else None,
        "START_POINTS": start_points
    }


def create_ngram_loglike_generation_data(model_id, ds_conta=None, model=None, tokenizer=None,
                                         n_starts=3, n=[30], random_starts=True,
                                         seed_rand=42, start_offset=5, only_problem=False,
                                         force_reprocess=False, verbose=False,
                                         data_name="MathCONTA",
                                         target_path=DRIVE_PATH / "MathCONTA_ngram_loglike_generation_data",
                                         base_path_token=DRIVE_PATH / "MathCONTA_tokens"):
    """

    """
    base_path = Path(target_path)
    cache_dir = base_path / model_id
    n_str = "_".join(map(str, n))
    file_path = cache_dir / f"{data_name}_nstarts{n_starts}_n{n_str}.json"
    print(f"File path: {file_path}")
    if file_path.exists() and not force_reprocess:
        print(f"Loading existing n-gram loglike generation data from: {file_path}")
        with open(file_path, 'r') as f:
            ngram_data = json.load(f)
        for entry in ngram_data:
            entry["NGRAM_LOGLIKE_GENERATION_DATA"] = {int(k): v for k, v in
                                                      entry["NGRAM_LOGLIKE_GENERATION_DATA"].items()}
        return ngram_data

    print("Processing n-gram loglike generation data from scratch...")
    ngram_data = []

    # Process each row in the dataset
    # get MathCONTAtokendata first
    MathCONTA_token_data = create_mathconta_token_data(model_id=model_id,
                                                       ds_conta=ds_conta,
                                                       model=model, tokenizer=tokenizer,
                                                       only_problem=False, force_reprocess=False,
                                                       base_path_token=base_path_token)

    for row in MathCONTA_token_data:
        print(f"Processing ID: {row['ID']}")

        if only_problem:
            problem_length = len(tokenizer.tokenize(row["PROBLEM"]))
            prompt_log = row["token_log_likes"][:problem_length]
            prompt_decoded = row["decoded_tokens"][:problem_length]
        else:
            prompt_log = row["token_log_likes"]
            prompt_decoded = row["decoded_tokens"]

        generation_results = {}

        # Iterate over each n-gram size in the list
        for n_val in n:
            generation_data = generate_ngram_loglike_data_row(
                input_loglikes=prompt_log,
                n_start=n_starts,
                n=n_val,
                random_starts=random_starts,
                start_offset=start_offset,
                seed_rand=seed_rand,
                verbose=verbose,
                prompt_decoded=prompt_decoded
            )
            generation_results[n_val] = generation_data

        # Combine the generation outputs with original dataset fields
        entry = {
            'ID': row['ID'],
            'CATEGORY': row['CATEGORY'],
            'PROBLEM': row['PROBLEM'],
            'SOLUTION': row['SOLUTION'],
            'LABEL': row['LABEL'],
            'LABEL_BINARY': row.get("LABEL_BINARY", None)
        }
        entry.update({'NGRAM_LOGLIKE_GENERATION_DATA': generation_results})
        ngram_data.append(entry)

    # Ensure the cache directory exists and then save the data.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ngram_data, f)

    print(f"Processed n-gram loglike generation data saved to: {file_path}")
    return ngram_data


def tune_ngram_loglike(ngram_data, feature_col, label_col, n_range, cv_folds, model_id, data_name, cv_seed, method_name,
                       log_path_base=Path(DRIVE_PATH / "not_specified"), exp_id='exp_id'):
    """

    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for n in n_range:
        # Get the DataFrame for current k
        ngram_df = get_ngram_loglike_df_from_generation_data(ngram_data, n)

        # Get the optimal threshold and CV metrics
        optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
            ngram_df, feature_col, label_col, cv_folds, random_state=cv_seed
        )

        # Format values: round to 6 decimal places and cast to native float
        median_threshold_fmt = round(float(statistics.median(optimal_thresholds)), 6)
        mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
        mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)
        theta_glob_fmt = round(float(theta_glob),6)

        # Print progress
        print(
            f"Testing n={n} | median_threshold={median_threshold_fmt:.6f} | mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

        # Log result
        log_entry = {
            "method": method_name,
            "parameter": {"n": n},
            "global_threshold": theta_glob_fmt,
            "median_threshold": median_threshold_fmt,
            "mean_cvacc_train": mean_cvacc_train_fmt,
            "mean_cvacc_test": mean_cvacc_test_fmt,
            "all_thresholds": [round(float(th), 6) for th in optimal_thresholds],
            "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
            "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
        }
        results.append(log_entry)

        # Update best result
        if mean_cvacc_test_fmt > best_score:
            best_score = mean_cvacc_test_fmt
            best_log_entry = log_entry

    # Print best log entry (main info only)
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "global_threshold": best_log_entry["global_threshold"],
        "median_threshold": best_log_entry["median_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'n': str(n_range), 'theta': "optimized"},
        "CV_folds": cv_folds,
        "CV_seed": cv_seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }
    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)
    return out_dict


def run_ngram_loglike(model_id, ds_conta, model, tokenizer, data_name,
                      force_reprocess=False, n_starts=1, n_range=[1],
                      seed_rand=42, start_offset=5, random_starts=True,
                      only_problem=False, verbose=False, dir_token_path=None, n=1, theta=0.5,
                      target_path=DRIVE_PATH / "MathCONTA_ngram_loglike_generation_data",
                      base_path_token=DRIVE_PATH / "MathCONTA_tokens"):
    """

    """
    # Step 1: Generate or load token-level data
    data = create_ngram_loglike_generation_data(model_id=model_id,
                                                ds_conta=ds_conta, data_name=data_name,
                                                model=model, tokenizer=tokenizer,
                                                n_starts=n_starts, n=n_range, random_starts=random_starts,
                                                seed_rand=seed_rand, start_offset=start_offset,
                                                only_problem=only_problem,
                                                force_reprocess=force_reprocess, verbose=verbose,
                                                target_path=target_path, base_path_token=base_path_token
                                                )

    # Step 2: Compute minK values into a DataFrame
    ngram_df = get_ngram_loglike_df_from_generation_data(data, n=n)

    # Step 3: Evaluate with provided theta
    all_metrics = run_on_ngram_loglike_df(ngram_df, theta=theta)

    # Step 4: Return selected metrics
    return all_metrics


def run_on_ngram_loglike_df(df, theta):
    """
    Compares cdd_value to threshold theta and evaluates prediction performance.

    Args:
        cdd_df (pd.DataFrame): DataFrame containing 'cdd_value' and 'LABEL_BINARY' columns.
        theta (float): Threshold for binarizing the cdd_value.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, confusion_matrix.
    """
    # Predict: 1 if value > theta, else 0 (all values are negative, values that are near 0 (aka higher) more risk of contamination)
    predictions = (df['ngram_loglike'] > theta).astype(int)

    # Ground truth
    ground_truth = df['LABEL_BINARY'].astype(int)

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
    }

# ===========================
# 5c. n_gram_cdd
# ===========================
def create_ngram_cdd_generation_data(model_id, model, tokenizer, torch_seed,
                                     n_starts, n_range, sample_size,
                                     random_starts, seed_rand, start_offset, verbose,
                                     ds_conta, only_problem,
                                     force_reprocess, data_name, base_path_generation):
    base_path = Path(base_path_generation)
    cache_dir = base_path / model_id
    n_str = "_".join(map(str, n_range))
    file_path = cache_dir / f"{data_name}_nstarts{n_starts}_n{n_str}_sample{sample_size}.json"
    print(f"File path: {file_path}")
    if file_path.exists() and not force_reprocess:
        print(f"Loading existing n-gram generation data from: {file_path}")
        with open(file_path, 'r') as f:
            ngram_data = json.load(f)
        for entry in ngram_data:
            entry["NGRAM_CDD_GENERATION_DATA"] = {int(k): v for k, v in entry["NGRAM_CDD_GENERATION_DATA"].items()}
        return ngram_data

    ngram_data = []

    # Set torch seed for reproducibility
    torch.manual_seed(torch_seed)
    if random_starts:
        random.seed(seed_rand)

    # Process each row in the dataset
    for row in ds_conta:
        print(f"Processing ID: {row['ID']}")

        # prepare prompt
        prompt = row["PROBLEM"] if only_problem else row["PROBLEM"] + row["SOLUTION"]

        # get starttokens list
        tokens = torch.tensor(tokenizer.encode(prompt))
        starttokens = select_start_points(tokens=tokens,
                                          n_start=n_starts,
                                          n=max(n_range),
                                          start_offset=start_offset,
                                          random_starts=random_starts,
                                          verbose=verbose)

        generation_results = generate_ngram_cdd_row(original_prompt=prompt,
                                                    starttokens=starttokens,
                                                    model=model, tokenizer=tokenizer,
                                                    n_range=n_range, sample_size=sample_size)

        # Combine the generation outputs with original dataset fields
        entry = {
            'ID': row['ID'],
            'CATEGORY': row['CATEGORY'],
            'PROBLEM': row['PROBLEM'],
            'SOLUTION': row['SOLUTION'],
            'LABEL': row['LABEL'],
            'LABEL_BINARY': row.get("LABEL_BINARY", None)
        }
        entry.update({'NGRAM_CDD_GENERATION_DATA': generation_results})
        ngram_data.append(entry)

    # Ensure the cache directory exists and then save the data.
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(ngram_data, f)

    print(f"Processed n-gram generation data saved to: {file_path}")

    return ngram_data


def generate_ngram_cdd_row(original_prompt, starttokens, model, tokenizer, n_range, sample_size):
    """
    Generates prompt snippets from the beginning of original_prompt up to each token in starttokens,
    then generates a cdd generation data row for each snippet using the maximum n from n_range.
    The generated output is then truncated for each n in n_range and grouped by n.

    Args:
        original_prompt (str): The full prompt text.
        starttokens (list[int]): A list of token indices where the prompt should be sliced.
        n_range (iterable of int): The range of n values (i.e., max_new_tokens) for which to obtain truncated outputs.
        model: The language model used for generation.
        tokenizer: The tokenizer corresponding to the model.
        sample_size (int): The number of samples to generate for each prompt snippet.

    Returns:
        dict: A dictionary where each key is an n value from n_range and the corresponding value is another dict with:
            - "starttoken": list of token indices (one for each prompt snippet).
            - "cdd_generation_data": list of cdd generation data dicts with keys "MAX_LENGTH", "GREEDY_IDS", "SAMPLE_IDS".
              The outputs are truncated to the first n tokens.
    """
    # Initialize the results dictionary with keys for each n in n_range.
    results = {n: {"starttoken": [], "cdd_generation_data": []} for n in n_range}

    # Determine the maximum n for full generation.
    max_n = max(n_range)

    for token_index in starttokens:
        # Create a prompt snippet from the start of the original prompt up to token_index.
        snippet = original_prompt[:token_index] if token_index <= len(original_prompt) else original_prompt

        # Generate full cdd generation data with max_new_tokens set to max(n_range)
        full_cdd_data = generate_cdd_data_row(
            prompt=snippet,
            model=model,
            tokenizer=tokenizer,
            sample_size=sample_size,
            max_new_tokens=max_n
        )

        # For each n in n_range, truncate the generated data accordingly and group the results.
        for n in n_range:
            # Truncate GREEDY_IDS to the first n tokens.
            greedy_ids_truncated = full_cdd_data["GREEDY_IDS"][:n]
            # Truncate each sample in SAMPLE_IDS to the first n tokens.
            sample_ids_truncated = [sample[:n] for sample in full_cdd_data["SAMPLE_IDS"]]
            # Compute the maximum length among truncated outputs.
            max_length_truncated = max(
                len(greedy_ids_truncated),
                max((len(sample) for sample in sample_ids_truncated), default=0)
            )

            truncated_data = {
                "GREEDY_IDS": greedy_ids_truncated,
                "SAMPLE_IDS": sample_ids_truncated,
                "MAX_LENGTH": max_length_truncated
            }

            # Append the token_index and the corresponding truncated data to the group for n.
            results[n]["starttoken"].append(token_index)
            results[n]["cdd_generation_data"].append(truncated_data)

    return results


def get_ngram_cdd_df_from_generation_data(ngram_generation_data, n, alpha):
    """

    """
    data = []

    for entry in ngram_generation_data:
        ngram_data_dict = entry.get("NGRAM_CDD_GENERATION_DATA", {})

        if n not in ngram_data_dict:
            available_ns = list(ngram_data_dict.keys())
            raise ValueError(
                f"Requested n={n} not found in NGRAM_CDD_GENERATION_DATA for ID {entry.get('ID', 'UNKNOWN')}. "
                f"Available n values: {available_ns}"
            )

        ngram_cdd_data = ngram_data_dict[n]

        peak_cdd_list = []
        for cdd_data_part in ngram_cdd_data["cdd_generation_data"]:
            max_length = cdd_data_part["MAX_LENGTH"]
            greedy_ids = cdd_data_part["GREEDY_IDS"]
            sample_ids = cdd_data_part["SAMPLE_IDS"]

            # Compute the token-level edit distance statistics
            rho_star_dict = calculate_rho_star(greedy_ids, sample_ids)
            peak_cdd = calculate_peak_cdd(rho_star_dict, max_length, alpha)
            peak_cdd_list.append(peak_cdd)

        ngram_cdd_mean = np.mean(peak_cdd_list)

        data.append({
            'ID': entry['ID'],
            'CATEGORY': entry['CATEGORY'],
            'LABEL': entry['LABEL'],
            'LABEL_BINARY': entry['LABEL_BINARY'],
            'n': n,
            'alpha': alpha,
            'ngram_cdd_mean': ngram_cdd_mean
        })

    return pd.DataFrame(data)


def tune_ngram_cdd(ngram_data, feature_col, label_col, n_range, alpha_range, cv_folds, model_id, data_name, cv_seed,
                   method_name,
                   log_path_base=Path(DRIVE_PATH / "not_specified"), exp_id='exp_id'):
    """

    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for n in n_range:
        for alpha in alpha_range:
            # Get the DataFrame for current k
            ngram_df = get_ngram_cdd_df_from_generation_data(ngram_data, n, alpha)

            # Get the optimal threshold and CV metrics
            optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
                ngram_df, feature_col, label_col, cv_folds, random_state=cv_seed
            )

            # Format values: round to 6 decimal places and cast to native float
            median_threshold_fmt = round(float(statistics.median(optimal_thresholds)), 6)
            mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
            mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)
            theta_glob_fmt = round(float(theta_glob), 6)

            # Print progress
            print(
                f"Testing n={n},alpha{alpha} | median_threshold={median_threshold_fmt:.6f} | mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

            # Log result
            log_entry = {
                "method": method_name,
                "parameter": {"n": n, "alpha": alpha},
                "global_threshold": theta_glob_fmt,
                "median_threshold": median_threshold_fmt,
                "mean_cvacc_train": mean_cvacc_train_fmt,
                "mean_cvacc_test": mean_cvacc_test_fmt,
                "all_thresholds": [round(float(th), 6) for th in optimal_thresholds],
                "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
                "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
            }
            results.append(log_entry)

            # Update best result
            if mean_cvacc_test_fmt > best_score:
                best_score = mean_cvacc_test_fmt
                best_log_entry = log_entry

    # Print best log entry (main info only)
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "global_threshold": best_log_entry["global_threshold"],
        "median_threshold": best_log_entry["median_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'n': str(n_range), 'theta': "optimized"},
        "CV_folds": cv_folds,
        "CV_seed": cv_seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }
    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)
    return out_dict


def run_ngram_cdd(model_id, model, tokenizer, torch_seed,
                  n_starts, n_range, sample_size,
                  random_starts, seed_rand, start_offset, verbose,
                  n, alpha, theta,
                  ds_conta, only_problem,
                  force_reprocess, data_name, base_path_generation):
    """

    """
    # Step 1: Generate or load token-level data
    data = create_ngram_cdd_generation_data(model_id=model_id,
                                            model=model,
                                            tokenizer=tokenizer,
                                            torch_seed=torch_seed,
                                            n_starts=n_starts,
                                            n_range=n_range,
                                            sample_size=sample_size,
                                            random_starts=random_starts,
                                            seed_rand=seed_rand,
                                            start_offset=start_offset,
                                            verbose=verbose,
                                            ds_conta=ds_conta,
                                            only_problem=only_problem,
                                            force_reprocess=force_reprocess,
                                            data_name=data_name,
                                            base_path_generation=base_path_generation)

    # Step 2: Compute minK values into a DataFrame
    ngram_df = get_ngram_cdd_df_from_generation_data(data, n=n, alpha=alpha)

    # Step 3: Evaluate with provided theta
    all_metrics = run_on_ngram_cdd_df(ngram_df, theta=theta)

    # Step 4: Return selected metrics
    return all_metrics


def run_on_ngram_cdd_df(df, theta):
    """
    """
    # Predict: 1 if value > theta, else 0 (all values are negative, values that are near 0 (aka higher) more risk of contamination)
    predictions = (df['ngram_cdd_mean'] > theta).astype(int)

    # Ground truth
    ground_truth = df['LABEL_BINARY'].astype(int)

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
    }

# ===========================
# 6. conta_traces (Contamination Traces)
# ===========================
#-ContaTraces-#
def get_ContaTraces_df_from_tokendata(list_data, fit="exponential"):
    """
    Processes a list of token log-likelihood data entries and fits them using the specified model,
    returning a DataFrame with extracted parameters and metadata.

    Args:
        list_data (list): A list of dictionaries, each containing token log-likelihood data along with
                          metadata fields: 'ID', 'CATEGORY', 'LABEL', and 'LABEL_BINARY'.
        fit (str, optional): The type of curve fitting to apply to the normalized cumulative log-likelihoods.
                             Supports "exponential" (default) and "linear". Defaults to "exponential".

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - ID (str): Identifier for the data entry.
            - CATEGORY (str): Category label.
            - LABEL (str): Original label.
            - LABEL_BINARY (int): Binary label (e.g., for classification).
            - fit (str): The type of fit used (e.g., "exponential" or "linear").
            - For exponential fit:
                - A_value (float): Fitted parameter A from the exponential model.
                - B_value (float): Fitted parameter B from the exponential model.
            - For linear fit:
                - m_value (float): Fitted slope m from the linear model y = -m*x.
    """
    data = []

    for list_element in list_data:
        # Calculate the cumulative log-likelihoods and normalize them.
        cum_loglikes = calc_cum_loglikes(list_element['token_log_likes'])
        cum_loglikes_norm = normalize_cum_loglikes(cum_loglikes)

        if fit == "exponential":
            # Fit using the exponential model and extract parameters.
            A_value, B_value, _ = fit_exponential(cum_loglikes_norm, plot=False)
            record = {
                'ID': list_element['ID'],
                'CATEGORY': list_element['CATEGORY'],
                'LABEL': list_element['LABEL'],
                'LABEL_BINARY': list_element['LABEL_BINARY'],
                'fit': fit,
                'A_value': A_value,
                'B_value': B_value
            }
        elif fit == "linear":
            # Use the outsourced linear fitting function.
            m_value = fit_linear(cum_loglikes_norm)
            record = {
                'ID': list_element['ID'],
                'CATEGORY': list_element['CATEGORY'],
                'LABEL': list_element['LABEL'],
                'LABEL_BINARY': list_element['LABEL_BINARY'],
                'fit': fit,
                'm_value': m_value
            }
        else:
            raise ValueError(f"Unknown fit type: {fit}")

        data.append(record)

    return pd.DataFrame(data)

#-ContaTraces-#
def calc_cum_loglikes(token_loglikes):
    """
    Computes cumulative log-likelihoods from a list of token log-likelihoods.

    Parameters:
        token_loglikes (list of float): Log-likelihoods of individual tokens.

    Returns:
        list of float: Cumulative log-likelihoods.
    """
    token_cum_loglikes = []

    for i in range(len(token_loglikes)):
        if i == 0:
            token_cum_loglikes.append(token_loglikes[i])
        else:
            token_cum_loglikes.append(token_loglikes[i] + token_cum_loglikes[i - 1])

    return token_cum_loglikes

#-ContaTraces-#
def normalize_cum_loglikes(cum_loglikes):
    """
    This function takes a list of cumulative log-likelihoods and normalizes them.

    input:
    cum_loglikes: list

    output:
    normalized_cum_loglikes: list
    """
    normalized_cum_loglikes = []
    n=len(cum_loglikes)
    first_value = cum_loglikes[0]
    for value in cum_loglikes:
        normalized_cum_loglike = (value - first_value)/n
        normalized_cum_loglikes.append(normalized_cum_loglike)

    return normalized_cum_loglikes

#-ContaTraces-#
def get_norm_cum_loglikes(input_text, model, tokenizer):
    """
    Compute normalized cumulative log-likelihoods and decode tokens from input text.

    Args:
        input_text (str): The text to process.
        model: The language model for inference.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        tuple: A pair (cum_loglikes_norm, decoded_tokens).
    """
    input_ids, logits = get_inference_logits(input_text, model, tokenizer)
    decoded_tokens, token_loglikes = get_token_loglikelihoods(tokenizer,logits, input_ids, False)
    cum_loglikes=calc_cum_loglikes(token_loglikes)
    cum_loglikes_norm = normalize_cum_loglikes(cum_loglikes)
    return cum_loglikes_norm, decoded_tokens

#-ContaTraces-#
def fit_exponential(normalized_cum_loglikes2, plot=False):
    """
    Fit the exponential model to the given data and return the parameters
    and fitted values. Optionally, plot the original data and the fitted curve.

    Parameters:
    - normalized_cum_loglikes2 (list or array): Data to fit the curve.
    - plot (bool): Whether to plot the original data and fitted curve.

    Returns:
    - a_fit (float): Fitted parameter A.
    - b_fit (float): Fitted parameter B.
    - fitted_y (list): Fitted y-values from the exponential model.
    """
    # Define the exponential function
    def exp_function(x, a, b):
        return (-1) * a * (1 - np.exp((-1) * b * x))

    # Generate indices as x-values
    indices = np.arange(len(normalized_cum_loglikes2))

    # Fit the curve
    try:
      popt, _ = curve_fit(exp_function, indices, normalized_cum_loglikes2, p0=(2, 0.01),maxfev=2000)
      a_fit, b_fit = popt
    except:
        return None, None, None

    # Generate fitted y-values
    fitted_y = exp_function(indices, a_fit, b_fit).tolist()

    # Optionally plot the data and fitted curve
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(indices, normalized_cum_loglikes2, 'o', label='Original Data',color='darkblue')
        plt.plot(indices, fitted_y, '-', label=f'Fitted Curve\nA={a_fit:.2f}, B={b_fit:.2f}',color='blue')
        plt.xlabel('Index')
        plt.ylabel('Normalized Cumulative Log-Likes')
        plt.title('Exponential Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    return a_fit, b_fit, fitted_y

#-ContaTraces-#
def fit_linear(cum_loglikes_norm):
    """
    Fits a linear model of the form y = m * x to the normalized cumulative log-likelihood data using curve_fit.

    Args:
        cum_loglikes_norm (array-like): The normalized cumulative log-likelihood values.

    Returns:
        float: The fitted slope parameter m. (will be < 0!)
    """

    # Define the linear model: y = m * x.
    def linear_model(x, m):
        return m * x

    # Generate x-axis values as the token indices.
    x = np.arange(len(cum_loglikes_norm))
    # Use curve_fit to determine the optimal m.
    popt, _ = curve_fit(linear_model, x, cum_loglikes_norm)
    return popt[0]

#-ContaTraces-#
def tune_ContaTraces(tokendata, feature_cols, label_col, fit_range, cv_folds, model_id, data_name, cv_seed, method_name,
                     log_path_base=Path("not_specified"), exp_id='exp_id'):
    """
    Tunes the minimum token frequency threshold (k) for a given dataset to optimize cross-validated accuracy.
    [Docstring remains unchanged...]
    """
    results = []
    best_log_entry = None
    best_score = -float('inf')

    for fit in fit_range:
        if fit == "exponential":
            A_value, B_value = feature_cols["exponential"]
            ContaTraces_df = get_ContaTraces_df_from_tokendata(tokendata, fit=fit)
            ContaTraces_df = clean_ContaTraces_df(ContaTraces_df)
            optimal_thresholds, train_accuracies, test_accuracies, theta_globs = find_optimal_2d_threshold_cv(
                ContaTraces_df, A_value, B_value, label_col,
                cv_folds=cv_folds, random_state=cv_seed, verbose=False
            )

            # Separate the two thresholds and compute medians
            thresholds_1 = [t[0] for t in optimal_thresholds]
            thresholds_2 = [t[1] for t in optimal_thresholds]
            median_threshold_1 = round(float(statistics.median(thresholds_1)), 6)
            median_threshold_2 = round(float(statistics.median(thresholds_2)), 6)
            # For exponential, theta_globs is a tuple
            theta_value = tuple(round(float(th), 6) for th in theta_globs)

        elif fit == "linear":
            # Assume single-threshold case
            feature_col = feature_cols["linear"]
            ContaTraces_df = get_ContaTraces_df_from_tokendata(tokendata, fit=fit)
            ContaTraces_df = clean_ContaTraces_df(ContaTraces_df)
            optimal_thresholds, train_accuracies, test_accuracies, theta_glob = find_optimal_threshold_from_df_cv(
                ContaTraces_df, "m_value", label_col,
                cv_folds=cv_folds, random_state=cv_seed, verbose=False
            )
            median_threshold_1 = round(float(statistics.median(optimal_thresholds)), 6)
            median_threshold_2 = None
            # For linear, theta_glob is a scalar
            theta_value = round(float(theta_glob), 6)

        else:
            raise ValueError(f"Unknown fit type: {fit}")

        # Format accuracy values
        mean_cvacc_train_fmt = round(float(statistics.mean(train_accuracies)), 6)
        mean_cvacc_test_fmt = round(float(statistics.mean(test_accuracies)), 6)

        # Print progress
        print(f"Testing fit={fit} | median_threshold_1={median_threshold_1:.6f} | "
              f"median_threshold_2={median_threshold_2 if median_threshold_2 is not None else 'N/A'} | "
              f"mean_cvacc_test={mean_cvacc_test_fmt:.6f}")

        # Log result including the theta value
        log_entry = {
            "method": method_name,
            "fit": fit,
            "parameter": {"fit": fit},
            "median_threshold_1": median_threshold_1,
            "median_threshold_2": median_threshold_2,
            "global_threshold": theta_value,
            "mean_cvacc_train": mean_cvacc_train_fmt,
            "mean_cvacc_test": mean_cvacc_test_fmt,
            "all_thresholds": [tuple(round(float(th), 6) for th in t) if isinstance(t, tuple) else round(float(t), 6)
                               for t in optimal_thresholds],
            "all_cvacc_train": [round(float(acc), 6) for acc in train_accuracies],
            "all_cvacc_test": [round(float(acc), 6) for acc in test_accuracies]
        }
        results.append(log_entry)

        # Update best result
        if mean_cvacc_test_fmt > best_score:
            best_score = mean_cvacc_test_fmt
            best_log_entry = log_entry

    # Print best log entry including the theta value
    print("Best log entry:")
    print({
        "parameter": best_log_entry["parameter"],
        "median_threshold_1": best_log_entry["median_threshold_1"],
        "median_threshold_2": best_log_entry["median_threshold_2"],
        "global_threshold": best_log_entry["global_threshold"],
        "mean_cvacc_train": best_log_entry["mean_cvacc_train"],
        "mean_cvacc_test": best_log_entry["mean_cvacc_test"]
    })

    # Build final output dictionary
    out_dict = {
        "model_id": model_id,
        "data_set": data_name,
        "datetime": datetime.now().isoformat(),
        "method_name": method_name,
        "parameter_range": {'fit': str(fit_range), 'theta': "optimized"},
        "CV_folds": cv_folds,
        "CV_seed": cv_seed,
        "best_log_entry": best_log_entry,
        "all_log_entries": results
    }

    # Save logs
    save_accuracy_log(log_path_base, model_id, method_name, exp_id, data_name, out_dict)

    return out_dict

#-ContaTraces-#
def clean_ContaTraces_df(df):
  df.fillna(0, inplace=True)
  return df

#-ContaTraces-#
def run_ContaTraces(model_id, ds_conta, model, tokenizer, dir_token_path,
                    fit="exponential", theta1=None, theta2=None,
                    only_problem=False, force_reprocess=False):
    """
    Full pipeline for ContaTraces-based classification evaluation on a dataset.

    This function:
      1. Loads or generates token-level data using `create_mathconta_token_data`.
      2. Computes curve-fit parameters from the token log-likelihoods by calling
         `get_ContaTraces_df_from_tokendata` with the specified fit ("exponential" or "linear").
      3. Compares the extracted parameters against provided thresholds to yield binary
         contamination predictions. For the exponential fit, the decision rule is:
             Predict 1 (contaminated) if A_value < theta1 OR B_value > theta2.
         For the linear fit, the decision rule is:
             Predict 1 if m_value >= theta1.
      4. Computes evaluation metrics (accuracy, precision, recall, f1_score, and confusion matrix)
         by comparing the predictions to the true binary labels.

    Parameters:
        model_id (str): Identifier for the model, also used for caching token data.
        ds_conta (datasets.Dataset): Hugging Face dataset containing required columns.
        model (PreTrainedModel): Hugging Face-compatible language model.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        dir_token_path (str or Path): Directory path for storing/loading cached token data.
        fit (str, optional): Curve fitting method to use: "exponential" (default) or "linear".
        theta1 (float): Primary threshold. For exponential fit, used on A_value; for linear, used on m_value.
        theta2 (float, optional): Secondary threshold required for exponential fit, used on B_value.
        only_problem (bool, optional): Whether to use only the problem text as model input. Defaults to False.
        force_reprocess (bool, optional): If True, reprocess token data even if cached data exists. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Overall classification accuracy.
            - 'precision': Precision of the positive (contaminated) class.
            - 'recall': Recall of the positive class.
            - 'f1_score': F1-score.
            - 'confusion_matrix': Confusion matrix as a numpy array.
    """
    if theta1 is None:
        raise ValueError("theta1 must be provided.")
    if fit == "exponential" and theta2 is None:
        raise ValueError("For exponential fit, theta2 must also be provided.")

    # Step 1: Generate or load token-level data.
    token_data = create_mathconta_token_data(
        model_id=model_id,
        ds_conta=ds_conta,
        model=model,
        tokenizer=tokenizer,
        only_problem=only_problem,
        force_reprocess=force_reprocess,
        base_path_token=dir_token_path
    )

    # Step 2: Compute ContaTraces DataFrame with curve-fit parameters.
    contaTraces_df = get_ContaTraces_df_from_tokendata(token_data, fit=fit)
    contaTraces_df = clean_ContaTraces_df(contaTraces_df)

    # Step 3: Evaluate predictions based on provided thresholds.
    metrics = run_ContaTraces_on_ContaTraces_df(contaTraces_df, fit, theta1, theta2)

    return metrics


#-ContaTraces-#
def run_ContaTraces_on_ContaTraces_df(df, fit, theta1, theta2=None):
    """
    Evaluates the ContaTraces DataFrame by applying a threshold decision rule on the extracted
    curve-fit parameters and computing classification metrics.

    For the "exponential" fit:
      - Uses columns 'A_value' and 'B_value'.
      - Decision rule: Predict 1 (contaminated) if A_value < theta1 OR B_value > theta2.

    For the "linear" fit:
      - Uses the 'm_value' column.
      - Decision rule: Predict 1 if m_value >= theta1.

    Parameters:
        df (pd.DataFrame): DataFrame containing the ContaTraces parameters and metadata.
        fit (str): The type of fit used: "exponential" or "linear".
        theta1 (float): Threshold for the primary parameter.
        theta2 (float, optional): Threshold for the secondary parameter (required for exponential fit).

    Returns:
        dict: Dictionary with evaluation metrics (accuracy, precision, recall, f1_score, confusion_matrix).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    if fit == "exponential":
        # Decision rule: contaminated if A_value < theta1 OR B_value > theta2.
        predictions = ((df['A_value'] < theta1) | (df['B_value'] > theta2)).astype(int)
    elif fit == "linear":
        # Decision rule: contaminated if m_value >= theta1.
        predictions = (df['m_value'] >= theta1).astype(int)
    else:
        raise ValueError(f"Unknown fit type: {fit}")

    ground_truth = df['LABEL_BINARY'].astype(int)

    # Compute evaluation metrics.
    acc = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
# ===========================
# 7. Visualization
# ===========================
