
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentraExtractor(nn.Module):

    def __init__(
        self,
        vocab_size=1989,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.30,
        padding_idx=1988
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict({
            "scenario_family": nn.Linear(256, 5),
            "intent_labels": nn.Linear(256, 9),
            "action": nn.Linear(256, 9),
            "target": nn.Linear(256, 56),
            "data_type": nn.Linear(256, 14),
            "destination": nn.Linear(256, 10),
            "authorization": nn.Linear(256, 3),
            "context_dependency": nn.Linear(256, 2)
        })

    def forward(self, input_ids, lengths):

        embeddings = self.embedding(input_ids)

        packed_embeddings = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden_state, _) = self.encoder(
            packed_embeddings
        )

        forward_state = hidden_state[-2]
        backward_state = hidden_state[-1]

        representation = torch.cat(
            [forward_state, backward_state],
            dim=1
        )

        representation = self.dropout(representation)

        outputs = {
            field: head(representation)
            for field, head in self.heads.items()
        }

        return outputs
