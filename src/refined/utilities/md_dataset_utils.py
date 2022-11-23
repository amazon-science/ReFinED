from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

cel_default_ignore_index = CrossEntropyLoss().ignore_index


def create_collate_fn(pad_id):
    """
    Collate function for mention (span) detection
    :param pad_id: value to use for padding
    :return: collate function (can be unsed in pytorch by `DataLoader` class)
    """

    def collate(items: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        b_max_seq_ln = max(x.size(0) for x, _ in items)
        b_size = len(items)
        b_token_ids = torch.zeros(size=(b_size, b_max_seq_ln), dtype=torch.long)
        b_token_ids.fill_(pad_id)
        b_labels = torch.zeros(size=(b_size, b_max_seq_ln), dtype=torch.long)
        # b_labels.fill_(-1)  # -1 used as ignore index
        b_labels.fill_(cel_default_ignore_index)  # -1 used as ignore index
        b_attention_mask = torch.zeros(size=(b_size, b_max_seq_ln), dtype=torch.float)
        for b_idx, (token_ids, labels) in enumerate(items):
            b_token_ids[b_idx, : token_ids.size(0)] = token_ids
            b_attention_mask[b_idx, : token_ids.size(0)] = 1.0
            b_labels[b_idx, : labels.size(0)] = labels
        return b_token_ids, b_attention_mask, b_labels

    return collate


def tokenize_and_preserve_labels(words, text_labels, tokenizer):

    if text_labels is None:
        text_labels = ['O' for _ in words]

    tokenized_sentence = []
    labels = []
    clean_words = []
    original_word_ix = []
    double_quote_count = 0
    prev_was_quote = False
    prev_token = None
    indices_to_filter = set()
    for word_idx, word in enumerate(words):
        if word_idx == 0:
            clean_words.append(word)
            continue
        if word == "," and word_idx + 2 < len(words) and words[word_idx + 1] == '"':
            indices_to_filter.add(word_idx)
        next_prev_token = None
        add_prefix_space = True
        if word in '"':
            double_quote_count += 1
            if double_quote_count % 2 == 0:
                add_prefix_space = False
                next_prev_token = '"END'

        if word in no_prefix_space_tokens:
            add_prefix_space = False

        if word == "s" and prev_was_quote:
            add_prefix_space = False

        if prev_token in {'"', "(", "[", "$", "£", "/"}:
            add_prefix_space = False

        if next_prev_token is not None:
            prev_token = next_prev_token
        else:
            prev_token = word
        word = " " + word if add_prefix_space else word
        prev_was_quote = word in {"'"}
        clean_words.append(word)

    assert len(clean_words) == len(
        text_labels
    ), f"token length ({len(words)}) must match labels length ({len(text_labels)})"
    for idx, (word, label) in enumerate(zip(clean_words, text_labels)):
        if idx in indices_to_filter:
            continue
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        original_word_ix.extend([idx for _ in tokenized_word])

        # Add the same label to the new list of labels n_subwords times
        if n_subwords > 1 and label[0] == "B":
            ner_type = "I" if len(label) == 1 else "I-" + label.split("-")[1]
            sub_labels = [label] + [ner_type] * (n_subwords - 1)
            labels.extend(sub_labels)
        else:
            labels.extend([label] * n_subwords)

    assert len(tokenized_sentence) == len(
        labels
    ), f"tokenized sentence does not align with labels {tokenized_sentence}"

    assert len(tokenized_sentence) == len(
        original_word_ix
    ), f"Must produce a mapping from each token to its position in the provided words"

    return tokenized_sentence[:], labels[:], original_word_ix


# list of tokens which should not have whitespace prepended
no_prefix_space_tokens = {
    "'",
    ")",
    ",",
    ".",
    "?",
    "!",
    ":",
    "'s",
    "n't",
    "'ve",
    "]",
    "'m",
    "'t",
    "'d",
    "'S",
    ";",
    "...",
    "/",
    "'re",
}


def bio_to_offset_pairs(bio_preds, use_labels: bool = False):

    offset_pairs = []
    in_ent = False
    current_start = None
    ent_type = None

    if use_labels:
        is_null = lambda x: x == 'O'
        is_start = lambda x: x[0] == 'B'
    else:
        is_null = lambda x: x == 0
        is_start = lambda x: x == 1

    for idx, bio_pred in enumerate(bio_preds):
        if not in_ent:
            # currently not in an entity
            if is_null(bio_pred):
                # do nothing
                pass
            elif is_start(bio_pred):
                # start ent
                current_start = idx
                in_ent = True
                ent_type = bio_pred.split('-')[-1] if use_labels else None
            else:
                # malformed by start ent
                current_start = idx
                in_ent = True
                ent_type = bio_pred.split('-')[-1] if use_labels else None

        else:
            # current in an entity
            if is_null(bio_pred):
                # end ent
                if use_labels:
                    offset_pairs.append((current_start, idx, ent_type))
                else:
                    offset_pairs.append((current_start, idx))
                in_ent = False
            elif is_start(bio_pred):
                # started new ent so finish current ent
                if use_labels:
                    offset_pairs.append((current_start, idx, ent_type))
                else:
                    offset_pairs.append((current_start, idx))
                current_start = idx
                ent_type = bio_pred.split('-')[-1] if use_labels else None
            else:
                # still in ent so do nothing
                pass
    if in_ent:
        if use_labels:
            offset_pairs.append((current_start, len(bio_preds), ent_type))
        else:
            offset_pairs.append((current_start, len(bio_preds)))
    return set(offset_pairs)
