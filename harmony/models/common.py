import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_length=10000):
        super(PositionalEncoding, self).__init__()

        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))

        # Pre-computing positional encoding values
        pe = torch.zeros(1, max_length, embedding_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward method to add positional encoding to an input tensor.
        The input tensor must have the following shape: (batch_size, seq_length, embedding_size)
        """
        assert x.size(1) <= self.max_length, f'Input sequence length ({x.size(1)}) must be less than ' \
                                             f'or equal to the max length ({self.max_length}).'

        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    """
    This module will be used when the input sequence is a long tensor with size (batch_size, seq_length),
    where each position is a long value, ranging from [0, vocab_size), indicating the word to integer mapping.
    """

    def __init__(self, vocab_size, embedding_size, dropout=0.1):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)

    def forward(self, x):
        """
        Forward method to apply an embedding layer to a sequence (embedding + positional encoding).
        The input tensor must have the following shape: (batch_size, seq_length), where each position is a long
        value, ranging from [0, vocab_size), indicating the word to integer mapping.
        """

        embedded = self.pos_encoding(self.embedding(x))
        return embedded


class LinearEmbeddingLayer(nn.Module):
    """
    This module will be used when the input sequence is a float tensor with size (batch_size, seq_length, in_features).
    """

    def __init__(self, in_features, embedding_size, dropout=0.1):
        super(LinearEmbeddingLayer, self).__init__()

        self.embedding = nn.Linear(in_features, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout)

    def forward(self, x):
        """
        Forward method to apply a linear embedding layer to a sequence (embedding + positional encoding).
        The input tensor must have the following shape: (batch_size, seq_length, in_features)
        """

        embedded = self.pos_encoding(self.embedding(x))
        return embedded

