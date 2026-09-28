import json
import re
from pathlib import Path

import torch
import torch.nn as nn

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# =========================================================
# NLTK resources
# =========================================================

for package in [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]:
    nltk.download(package, quiet=True)


# =========================================================
# Paths + artifacts
# =========================================================

ART = Path(__file__).parent / "artifacts"

with open(ART / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

with open(ART / "word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)


SEQ_LEN = int(config["seq_len"])
VOCAB_SIZE = int(config["vocab_size"])
EMBEDDING_DIM = int(config["embedding_dim"])
HIDDEN_SIZE = int(config["hidden_size"])
CLASSES = config["classes"]


# =========================================================
# EXACT preprocessing used in the notebook
# =========================================================

wnl = WordNetLemmatizer()

mystopwords = set(stopwords.words("english"))

# Keep negation words exactly like training
for word in ["not", "no", "nor"]:
    if word in mystopwords:
        mystopwords.remove(word)


def clean(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"\brt\b",
        " ",
        text
    )

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = word_tokenize(text)

    textlist = []

    for term in tokens:

        if term in mystopwords:
            continue

        if len(term) < 2:
            continue

        if not term.replace("'", "").isalpha():
            continue

        term = wnl.lemmatize(term)

        textlist.append(term)

    return textlist


# =========================================================
# EXACT encoding used during training
# =========================================================

def encode_text(tokens):

    sequence = []

    for word in tokens:

        if word in word2idx:
            sequence.append(
                word2idx[word]
            )

        else:
            sequence.append(
                word2idx["<OOV>"]
            )

    # Training keeps the LAST 40 tokens
    sequence = sequence[-SEQ_LEN:]

    # Training LEFT-pads with zeros
    while len(sequence) < SEQ_LEN:
        sequence.insert(0, 0)

    return sequence


# =========================================================
# Model — same architecture as notebook
# =========================================================

class SimpleRNNModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_size=144,
        num_classes=3
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(
            0.15
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh"
        )

        self.fc1 = nn.Linear(
            hidden_size,
            64
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            0.30
        )

        self.fc2 = nn.Linear(
            64,
            num_classes
        )


    def forward(self, x):

        x = self.embedding(x)

        x = self.embedding_dropout(x)

        out, hidden = self.rnn(x)

        x = hidden[-1]

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# =========================================================
# Load trained model
# =========================================================

device = torch.device("cpu")

model = SimpleRNNModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=len(CLASSES)
)

state_dict = torch.load(
    ART / "best_simple_rnn.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

model.to(device)

model.eval()


# =========================================================
# Prediction
# =========================================================

def predict_sentiment(text):

    tokens = clean(text)

    sequence = encode_text(tokens)

    x = torch.tensor(
        [sequence],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


    scores = {
        CLASSES[i]: float(
            probabilities[i].item()
        )
        for i in range(len(CLASSES))
    }


    best_index = int(
        torch.argmax(probabilities).item()
    )

    label = CLASSES[best_index]


    return label, scores