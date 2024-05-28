import torch
import torch.nn as nn

from .common import EmbeddingLayer
from .common import LinearEmbeddingLayer


class TransformerEncoder(nn.Module):
    def __init__(self, in_features, embedding_size, num_heads, dim_feedforward, num_layers=1, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(embedding_size, num_heads, dim_feedforward,
                                                   dropout, activation='gelu', batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embedding = LinearEmbeddingLayer(in_features, embedding_size, dropout)

    def forward(self, x, mask=None, padding_mask=None):
        """
        Forward method to pass a sequence through a Transformer Encoder block.
        The input tensor must have the following shape: (batch_size, sequence_length, embedding_size).

        Optionally, you can inform an attention mask and a padding mask, to mask padding input tokens.
        """
        embedded = self.embedding(x)
        outputs = self.encoder(embedded, mask, padding_mask)

        return outputs


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers=1, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(embedding_size, num_heads, dim_feedforward,
                                                   dropout, activation='gelu', batch_first=True)

        self.fc = nn.Linear(embedding_size, vocab_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.embedding = EmbeddingLayer(vocab_size, embedding_size, dropout)

    def forward(self, x, memory, mask=None):
        """
        Forward method to pass a sequence through a Transformer Decoder block.
        The input tensor must have the following shape: (batch_size, seq_length), where each position is a long value, ranging
        from [0, vocab_size), indicating the word to integer mapping. Additionally, you must inform a memory tensor, which is the
        output of the last Transformer Encoder layer, having the following shape: (batch_size, seq_length, embedding_size)

        Optionally, you can inform an attention causal mask, where -inf represent unattended attention tokens.
        """

        embedded = self.embedding(x)
        outputs = self.decoder(embedded, memory, mask)
        outputs = self.fc(outputs)

        return outputs


class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, padding_mask=None):
        """
        Forward method to pass a sequence through a sequence-to-sequence Transformer.
        The src tensor must have the following shape: (batch_size, sequence_length, in_features), while the tgt tensor has the shape
        (batch_size, seq_length), where each position is a long value, ranging from [0, vocab_size), indicating the word to integer mapping.

        Optionally, you can inform an attention causal mask for the target, where -inf represent unattended attention tokens, a source
        attention mask and a source padding mask.
        """
        memory = self.encoder(src, src_mask, padding_mask)
        outputs = self.decoder(tgt, memory, tgt_mask)

        return outputs

    # def inference(self, src, sos_index, src_mask=None, padding_mask=None):
    #     """
    #     Method to make a forward pass of the sequence-to-sequence transformer without using the target tensor as input, i.e. the prediction
    #     is made with the transformer's own previous predictions rather than using a teacher forcing mechanism.
    #
    #     The src tensor must have the following shape: (batch_size, sequence_length, in_features), and you must provide de <sos> token index.
    #
    #     Optionally, you can inform a source attention mask and a source padding mask.
    #     """
    #     batch_size = src.size(0)
    #     memory = self.encoder(src, src_mask, padding_mask)
    #     decoder_input = torch.full((batch_size, 1), fill_value=sos_index)
    #
    #     outputs = []
    #     MAX_DECODING_LENGTH = 64
    #
    #     for i in range(MAX_DECODING_LENGTH):
    #         output = self.decoder(decoder_input, memory)
    #         last_output = output[:, -1].unsqueeze(1)
    #
    #         outputs.append(last_output)
    #         prediction = torch.argmax(last_output, dim=-1)
    #         decoder_input = torch.cat([decoder_input, prediction], dim=1)
    #
    #     outputs = torch.cat(outputs, dim=1)
    #     return outputs