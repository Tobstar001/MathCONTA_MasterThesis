import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
import json
from datetime import datetime
import pytz
import itertools
from itertools import cycle
import re

## CONSTANTS

INPUT_MAPPINGS = {
    "ngram_acc": lambda row: row["Q"] + " " + row["Agold"],
    "cdd": lambda row: row["Q"],
    "minK": lambda row: row["Q"] + " " + row["Agold"],
    "conta_traces": lambda row: row["Q"] + " " + row["Agold"],
}

## --------------------------- CDD --------------------------- ##

# Function to calculate token-level edit distance
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
        greedy_output: The greedy output token sequence.
        output_list: A list of generated output token sequences.

    Returns:
        A dictionary where keys are edit distances (d) and values are ρ*(d).
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
        rho_star: A dictionary where keys are edit distances (d) and values are ρ*(d).
        alpha: The threshold for inclusion in the peak CDD calculation.

    Returns:
        The peak cumulative diversity density.
    """
    threshold = alpha * max_length
    peak_cdd = sum(value for key, value in rho_star.items() if key <= threshold)
    return peak_cdd

# CDD pipeline function
def run_cdd(prompt, model, tokenizer, sample_size, alpha=0.05, zeta=0.01, verbose=False):
    """
    Assesses the diversity of a language model's generated outputs using Cumulative Diversity Density (CDD).

    Args:
        prompt (str): The input prompt for the language model.
        model: The pre-trained language model to use for generation.
        tokenizer: The tokenizer associated with the language model.
        sample_size (int): The number of outputs to generate for analysis.
        alpha (float, optional): The threshold for inclusion in peak CDD calculation. Defaults to 0.05.
        zeta (float, optional): The threshold for determining if CDD is significant. Defaults to 0.01.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        bool: True if the peak CDD exceeds the threshold (zeta), indicating potential contamination, False otherwise.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    greedy_output = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=30, do_sample=False)
    generated_tokens = greedy_output[:, inputs["input_ids"].shape[-1]:]
    greedy_output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    if verbose:
        print("Greedy Output:")
        print(greedy_output_text)

    output_list = []
    max_length = len(greedy_output_text)
    for i in range(sample_size):
        output = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=30, do_sample=True)
        generated_tokens = output[:, inputs["input_ids"].shape[-1]:]
        output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        output_list.append(output_text)

        if verbose:
            print(f"Sample {i + 1} Output:")
            print(output_text)

        if len(output_text) > max_length:
            max_length = len(output_text)

    rho_star_dict = calculate_rho_star(greedy_output_text, output_list)
    peak_cdd = calculate_peak_cdd(rho_star_dict, max_length, alpha)

    if verbose:
        print("\nRho Star Dictionary:")
        print(rho_star_dict)
        print("\nPeak CDD:")
        print(peak_cdd)

    output = peak_cdd > zeta

    return output, peak_cdd

## --------------------------- minK --------------------------- ##
def get_inference_logits(sentence, model, tokenizer):
    """
    This function takes a sentence and returns the logits for each token in the sentence.

    input:
    sentence: str
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer

    output:
    input_ids: torch.Tensor (torch.Size([1, len(tokens)]))
    logits: torch.Tensor (torch.Size([1, len(tokens), vocab_size])
    """
    tokens = tokenizer(sentence, return_tensors="pt")

    input_ids = tokens["input_ids"] #torch.Size([1, 10])

    # Move input tokens (=tensors) to GPU (if available)
    tokens = {key: value.to("cuda") for key, value in tokens.items()}

    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**tokens)
    logits = outputs.logits #torch.Size([1, 10, 128256])

    return input_ids, logits

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

def get_token_loglikelihoods(tokenizer, logits, input_ids, debug=False):
    """
    This function takes logits and input_ids and returns the log_likelihood for each token. (alternative approach!!)

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
            print(f"Token: {token}, Probability: {prob:.4f}")

    return decoded_tokens, token_logs

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

## --------------------------- conta_traces --------------------------- ##

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



def run_contraces(input_text, model, tokenizer, a_delta=2, b_delta=0.01, verbose=False):
    """
    Runs the Contraces
    """
    cum_loglikes_norm, _ = get_norm_cum_loglikes(input_text, model, tokenizer)

    a, b, _ = fit_exponential(cum_loglikes_norm, False)
    if a is None or b is None:
        return None, (None, None)
    if a<a_delta:
      return True, (a,b)
    elif b>b_delta:
      return True, (a,b)
    else:
      return False, (a,b)

## plot functions ##
#group plot
def plot_line_chart_with_labels(y_values_list, labels, x_tick_labels, x_label, y_label, title):
    """
    Plots a line chart with multiple lines grouped by two categories, with a single legend entry per group.

    Parameters:
    y_values_list (list): A list of lists containing the y-values for each line.
    labels (list): A list of strings representing the group for each line.
    x_tick_labels (list): A list of strings representing the x-tick labels (can be None).
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))

    num_lines = len(y_values_list)
    line_width = max(1, 3 - num_lines)  # Adjust line width: thinner for more lines

    # Use a color cycle for two classes of labels
    group_colors = {'Clean': 'tab:blue', 'Conta': 'tab:orange'}
    label_group_map = {label: 'Clean' if label in [0] else 'Conta' for label in labels}

    # Track plotted groups to avoid duplicate legend entries
    plotted_groups = set()

    # Plot each line with group colors
    for y_values, label in zip(y_values_list, labels):
        group = label_group_map[label]
        color = group_colors[group]

        # Plot line with group color
        l = len(y_values)
        marker = 'o' if num_lines <= 3 else None  # Discard markers for more than 3 lines
        plt.plot(
            list(range(l)),
            y_values,
            label=group if group not in plotted_groups else None,  # Avoid duplicate legend entries
            linewidth=line_width,
            marker=marker,
            color=color
        )

        plotted_groups.add(group)

    # Rotate x-axis tick labels
    if x_tick_labels:
        plt.xticks(list(range(len(x_tick_labels))), labels=x_tick_labels, rotation=90)

    # Add labels, title, and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    # Adjust layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()

#versus plot
def plot_line_chart_dual(y_values_list, labels, x_tick_labels_bottom, x_tick_labels_top, x_label, y_label, title):
    """
    Plots a line chart with multiple lines, applying specific colors for designated labels,
    bottom x-tick labels rotated 90 degrees, and an additional x-tick label row on the top.

    Parameters:
        y_values_list (list): A list of lists containing the y-values for each line.
        labels (list): A list of strings representing the labels for each line.
        x_tick_labels_bottom (list): A list of strings representing the bottom x-tick labels.
        x_tick_labels_top (list): A list of strings representing the top x-tick labels.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(24, 8))

    # Define a mapping from label to color
    color_map = {
        0: "C0",
        1: "C1"
    }

    # Plot each line, using a specific color if defined in color_map
    for y_values, label in zip(y_values_list, labels):
        l = len(y_values)
        color = color_map.get(label)  # Use None if label not in the dictionary
        plt.plot(list(range(l)), y_values, label=label, marker='o', color=color)

    # Bottom x-ticks and legend
    plt.xticks(list(range(len(x_tick_labels_bottom))), labels=x_tick_labels_bottom, rotation=90)
    plt.legend()

    # Create a twin x-axis for the top x-tick labels
    ax = plt.gca()
    ax_top = ax.twiny()
    ax_top.set_xticks(list(range(len(x_tick_labels_top))))
    ax_top.set_xticklabels(x_tick_labels_top, rotation=90)
    ax_top.set_xlim(ax.get_xlim())  # Align the top x-axis with the bottom

    # Add axis labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Adjust layout for a better fit
    plt.tight_layout()
    plt.show()

#needed for versus plot as some characters cannot be printed :(
def clean_list_strings(strings):
    """
    Remove non-alphanumeric characters from each string in a list.

    This function iterates over each element in the provided list,
    replacing any character that is not a letter or digit with an empty string.

    Args:
        strings (list of str): The list of strings to clean.

    Returns:
        list of str: A new list with cleaned strings.
    """
    return [re.sub(r'[^A-Za-z0-9]', '*', s) for s in strings]

## --------------------------- perplexity (Xu) --------------------------- ##
def atomic_perplexity(sentence, model, tokenizer):
  """
  Calculates the atomic perplexity of a sentence using a pre-trained language model.

  Parameters:
    sentence (str): The input sentence for perplexity calculation.
    model (transformers.PreTrainedModel): The pre-trained language model.
    tokenizser (transformers.PreTrainedTokenizer): The tokenizer associated with the model.

  Returns:
    float: The atomic perplexity of the sentence.
  """
  input_ids, logits = get_inference_logits(sentence, model, tokenizer)
  decoded_tokens, token_loglikes = get_token_loglikelihoods(tokenizer, logits, input_ids, False)
  cum_loglikes=calc_cum_loglikes(token_loglikes)
  t = len(cum_loglikes)
  ppl = np.exp(-1/t * cum_loglikes[-1])
  return ppl


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

def run_minK(input_minK, model, tokenizer, k, delta, verbose=False):
    """

    :param input_minK: text which incorporates Q + Agold
    :param model:
    :param tokenizer:
    :param k: value between 0 and 100 (how many tokens percentage-wise should be in the minK set)
    :param delta: minK must be higher than delta so it is flaged as contaminated
    :param verbose: Print the tokens with their probability
    :return: output, minK value

    """
    input_ids, logits = get_inference_logits(input_minK, model, tokenizer)
    decoded_tokens, token_probs = get_token_probabilities(tokenizer, logits, input_ids, verbose)
    minK = get_mink_value(token_probs, k)

    output = minK > delta

    return output, minK

## --------------------------- ngram accuracy --------------------------- ##

def n_gram_accuracy(token_sequence, model, tokenizer, n=3, sample_size=10, verbose=False):
    """
    Computes the N-gram Accuracy for a given token sequence.

    Parameters:
        token_sequence (str): The input text sequence.
        model: The transformer model used for predictions.
        tokenizer: The tokenizer associated with the transformer model.
        n (int): The size of the N-gram. Default is 3.
        sample_size (int): The number of random starting points. Default is 10.
        verbose (bool): If True, prints prompt, predicted, and actual N-grams. Default is False.

    Returns:
        float: The N-gram Accuracy.
    """
    # Tokenize the input sequence
    inputs = tokenizer(token_sequence, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}  # Move tokens and attention mask to CUDA
    tokens = inputs["input_ids"].squeeze()
    token_length = tokens.size(0)

    # Ensure there are enough tokens to sample
    if token_length <= n:
        raise ValueError("Token sequence is too short for the given N-gram size.")

    # Generate K random starting points
    start_points = random.sample(range(1,token_length - n), sample_size)
    correct_predictions = 0

    for start in start_points:
        # Create a copy of inputs to avoid modifying the original dictionary
        current_inputs = copy.deepcopy(inputs)

        # Define the prompt and target N-gram
        current_inputs["input_ids"] = tokens[:start].unsqueeze(0)
        if "attention_mask" in current_inputs:
            current_inputs["attention_mask"] = current_inputs["attention_mask"][:, :start]

        target_ngram = tokens[start:start + n]

        # Generate model predictions
        with torch.no_grad():
            outputs = model.generate(**current_inputs, max_new_tokens=n,pad_token_id=tokenizer.eos_token_id)

        predicted_ngram = outputs[0, -n:]

        # Compare predicted N-gram to the true N-gram
        if torch.equal(predicted_ngram, target_ngram):
            correct_predictions += 1

        if verbose:
            # Decode and print details
            prompt_text = tokenizer.decode(tokens[:start], skip_special_tokens=True)
            predicted_text = tokenizer.decode(predicted_ngram, skip_special_tokens=True)
            actual_text = tokenizer.decode(target_ngram, skip_special_tokens=True)
            print(f"Prompt: {prompt_text}\nPredicted: {predicted_text}\nActual: {actual_text}\n")

    # Compute N-gram accuracy
    n_gram_accuracy = correct_predictions / sample_size
    return n_gram_accuracy

def run_ngram_accuracy(input_text, model, tokenizer, n, sample_size, delta, verbose=False):
    """
    Evaluates the N-gram Accuracy for a given input text and determines if it surpasses a specified threshold.

    Parameters:
        input_text (str): The text sequence to be evaluated.
        model: The transformer model used for predictions.
        tokenizer: The tokenizer associated with the transformer model.
        n (int): The size of the N-gram.
        sample_size (int): The number of random starting points for evaluation.
        delta (float): A threshold between 0 and 1; if the proportion of correct N-grams exceeds this value, a contamination flag is raised.
        verbose (bool, optional): If True, prints details of the predicted and actual N-grams. Default is False.

    Returns:
        tuple:
            - bool: True if N-gram accuracy surpasses the delta threshold, otherwise False.
            - float: The computed N-gram Accuracy.
    """
    n_gram_acc = n_gram_accuracy(input_text,model,tokenizer,n,sample_size,verbose)
    output = n_gram_acc>delta
    return output, n_gram_acc


### --------------------------END2END Pipeline-----------------------------###

def tune_method_parameters(selected_method_name, METHODS, PARAMETER_GRID, DRIVE_PATH, ds, experiment_id, model_id,
                           data_id, data_version, seed, model, tokenizer, INPUT_MAPPINGS, verbose=False,
                           log_verbose=False):
    """
    Runs a parameter tuning experiment for a given method and logs the results.

    Parameters:
    - selected_method_name: str, name of the method to test
    - METHODS: dict, mapping of method names to functions
    - PARAMETER_GRID: dict, mapping method names to parameter grids
    - DRIVE_PATH: Path, base directory to save logs
    - ds: iterable, dataset containing input rows
    - experiment_id, model_id, data_id, data_version, seed: experiment metadata
    - model, tokenizer: model components used in testing
    - INPUT_MAPPINGS: dict, mapping method names to input transformation functions
    - verbose: bool, whether to print general logs
    - log_verbose: bool, whether to print detailed log entries

    Returns:
    - best_parameters: dict, parameter set with the highest accuracy
    - best_score: float, best accuracy achieved
    """
    selected_method = METHODS[selected_method_name]
    filename = f"exp_{experiment_id}_{selected_method_name}.json"

    # Generate all parameter combinations for tuning
    parameter_combinations = list(itertools.product(*PARAMETER_GRID[selected_method_name].values()))
    parameter_keys = list(PARAMETER_GRID[selected_method_name].keys())

    # Initialize storage for best parameters
    best_score = float("-inf")
    best_parameters = None
    accuracy_results = []  # List to store accuracy results

    # Define file paths
    experiment_dir = DRIVE_PATH / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = experiment_dir / filename
    accuracy_file_path = experiment_dir /"accuracy_log.json"

    with open(log_file_path, "a") as log_file:
        for params in parameter_combinations:
            # Convert tuple to dictionary
            parameters = dict(zip(parameter_keys, params))
            correct_predictions = 0
            total_predictions = 0

            if verbose:
                print(f"Testing parameters: {parameters} on method {selected_method_name}")

            for row in ds:
                # Initialize log entry
                dictlog_single = {
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "data_id": data_id,
                    "data_version": data_version,
                    "seed": seed,
                    "ID": row["ID"],
                    "label": row["label"],
                    "timestamp": datetime.now(pytz.timezone("Europe/Paris")).isoformat(),
                    "method_name": selected_method_name,
                    "parameters": parameters,
                }

                # Get the correct input format for the selected method
                input_text = INPUT_MAPPINGS[selected_method_name](row)

                # seed
                torch.manual_seed(seed)

                # Run method with the current parameter set
                method_output = selected_method(input_text, model, tokenizer, verbose=False, **parameters)

                # Ensure output is properly formatted
                if isinstance(method_output, tuple):
                    output_data, metric_value = method_output
                else:
                    output_data = method_output
                    metric_value = None  # Default if there's no second value returned

                # Compute accuracy: Compare method output to the label
                prediction_correct = int(output_data == row["label"])
                correct_predictions += prediction_correct
                total_predictions += 1

                # Update log with metrics
                metrics = {"output": output_data, selected_method_name: metric_value}
                dictlog_single.update(metrics)

                if log_verbose:
                    print(dictlog_single)

                # Write log entry to file
                if log_file.tell() != 0:
                    log_file.write("\n")  # Add newline if it's not the first entry
                json.dump(dictlog_single, log_file, indent=None)

            # Compute accuracy for this parameter set
            accuracy = correct_predictions / total_predictions
            if verbose:
                print(f"Accuracy for parameters {parameters}: {accuracy:.4f}")

            # Store accuracy result
            accuracy_results.append({
                "method_name": selected_method_name,
                "parameters": parameters,
                "accuracy": accuracy
            })

            # Update best parameter set if the new one is better
            if accuracy > best_score:
                best_score = accuracy
                best_parameters = parameters

    # Save accuracy results to a separate file
    accuracy_log = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "data_id": data_id,
        "data_version": data_version,
        "seed": seed,
        "best_parameters": best_parameters,
        "best_accuracy": best_score,
        "all_results": accuracy_results
    }

    with open(accuracy_file_path, "a") as acc_file:
        json.dump(accuracy_log, acc_file, indent=4)

    print(f"Best Parameters: {best_parameters} with accuracy: {best_score:.4f}")
    print(f"Accuracy log saved in {accuracy_file_path}")

    return best_parameters, best_score


### --------------------------EVALUATION-----------------------------###

def read_and_evaluate_linewise_json(file_path):
    """
    Reads a JSON file containing multiple JSON objects (one per line),
    evaluates the 'output' against the 'label', and calculates accuracy.

    :param file_path: Path to the JSON file
    :return: Evaluation metrics (e.g., accuracy)
    """
    correct_predictions = 0
    total_samples = 0

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip any empty lines
                entry = json.loads(line)
                if entry["output"] == entry["label"]:
                    correct_predictions += 1
                total_samples += 1

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    return {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy
    }
