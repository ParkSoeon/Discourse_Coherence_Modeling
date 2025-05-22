"""
Description:
    * Read each line at once
    * Split by Spacing by lines 
        ex) ["나는", "학생이다"], ["O", "B_OG"]
    * Print out the warning message if numbers of Tokens and Tags are different
    '''
        tokens = [
        ["나는", "학생이다"],
        ["서울", "대", "출신"],
        ...
        ]
        labels = [
            ["O", "B_OG"],
            ["B_LC", "I_LC", "O"],
            ...
        ]
    '''
"""
def load_data(words_path, tags_path):
    tokens, labels = [], []
    with open(words_path, "r", encoding = "utf-8") as wf, open(tags_path, "r", encoding = "utf-8") as tf:
        for w_line, t_line in zip(wf, tf):
            w = w_line.strip().split()
            t = t_line.strip().split()

            if len(w) != len(t):
                print(f"[WARNING] Tokan, Tag Length Mismatched: {len(w)} != {len(t)}")

            tokens.append(w)
            labels.append(t)

    return tokens, labels

# def align_labels_with_tokens(example, tokenizer, label2id):
#     tokenized_inputs = tokenizer(
#         example["tokens"],
#         truncation=True,
#         is_split_into_words=True
#     )

#     word_ids = tokenized_inputs.word_ids()
#     labels = []

#     word_labels = example["labels"]
#     if isinstance(word_labels[0], list):
#         word_labels = word_labels[0]

#     previous_word_idx = None
#     for word_idx in word_ids:
#         if word_idx is None:
#             labels.append(-100)
#         elif word_idx != previous_word_idx:
#             labels.append(label2id.get(word_labels[word_idx], -100))
#         else:
#             labels.append(-100)

#         previous_word_idx = word_idx

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

"""
Description:
    
"""
def align_labels_with_tokens(example, tokenizer, label2id):
    tokens = example["tokens"]
    labels = example["labels"]

    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors=None,  
        truncation=True,
        padding=False,        
    )

    word_ids = tokenized.word_ids()
    aligned_labels = []

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        else:
            aligned_labels.append(label2id.get(labels[word_idx], -100))

    tokenized["labels"] = aligned_labels
    return tokenized

def align_labels_with_tokens(example, tokenizer, label2id):
    tokens = example["tokens"]
    labels = example["labels"]

    tokenized = tokenizer(
        tokens,
        is_split_into_words = True,
        return_offsets_mapping = True,
        return_tensors = None, 
        truncation = True,
        padding = False
    )