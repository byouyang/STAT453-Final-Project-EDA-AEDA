"""EDA reference architecture: BiLSTM(64) -> dropout -> BiLSTM(32) -> dropout."""

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from .config import GLOVE_DIM, HIDDEN_UNITS, LSTM_DROPOUT, LSTM_HIDDEN_SIZES, NUM_CLASSES


class TextLSTM(nn.Module):
    """Concatenate the second layer's final forward and backward states.

    Lengths are derived from token IDs, not vector values: unknown words have
    zero embeddings in the CNN protocol but are still real sequence tokens.
    Empty inputs use a zero sentence representation.
    """

    def __init__(self, embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != GLOVE_DIM:
            raise ValueError(f"Expected a [vocab, {GLOVE_DIM}] embedding matrix.")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(), freeze=True, padding_idx=0
        )
        first_size, second_size = LSTM_HIDDEN_SIZES
        self.lstm1 = nn.LSTM(GLOVE_DIM, first_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(LSTM_DROPOUT)
        self.lstm2 = nn.LSTM(2 * first_size, second_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(LSTM_DROPOUT)
        self.hidden = nn.Linear(2 * second_size, HIDDEN_UNITS)
        self.relu = nn.ReLU()
        self.output = nn.Linear(HIDDEN_UNITS, NUM_CLASSES)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        lengths = token_ids.ne(0).sum(dim=1)
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(
            embedded, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        sequence, _ = self.lstm1(packed)
        # Drop outputs between layers, as in Keras Dropout(0.5). Preserve packed
        # ordering/lengths so both directions ignore right-padding.
        sequence = PackedSequence(
            self.dropout1(sequence.data), sequence.batch_sizes,
            sequence.sorted_indices, sequence.unsorted_indices,
        )
        _, (hidden, _) = self.lstm2(sequence)
        sentence = torch.cat((hidden[0], hidden[1]), dim=1)
        sentence = sentence.masked_fill(lengths.eq(0).unsqueeze(1), 0.0)
        return self.output(self.relu(self.hidden(self.dropout2(sentence))))
