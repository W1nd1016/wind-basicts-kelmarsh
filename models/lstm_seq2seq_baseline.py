import torch
import torch.nn as nn


class LSTMSeq2SeqBaseline(nn.Module):
    """
    Official-style Seq2Seq LSTM baseline for wind farm forecasting.

    Encoder input:  X (B, L, N, F)  (use all F, typically 7 for only-scada)
    Decoder: predicts power autoregressively for H steps:
        y_hat[t] depends on hidden state and previous y (teacher forcing in training)

    Output: (B, H, N)
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.N = num_nodes
        self.F = input_dim
        self.H = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encode ALL nodes jointly: (B, L, N, F) -> (B, L, N*F)
        self.enc_in = nn.Linear(num_nodes * input_dim, hidden_dim)

        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder takes previous power vector (B, N) as input
        self.decoder_cell = nn.LSTMCell(
            input_size=num_nodes,   # previous y (all nodes)
            hidden_size=hidden_dim,
        )

        self.out_proj = nn.Linear(hidden_dim, num_nodes)  # -> y_t for all nodes

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, X, teacher_forcing_y=None, teacher_forcing_ratio: float = 0.0):
        """
        X: (B, L, N, F)
        teacher_forcing_y: (B, H, N) ground truth future (normalized), optional
        teacher_forcing_ratio: probability to use gt as previous y during decoding
        """
        B, L, N, F = X.shape
        assert N == self.N and F == self.F

        # -------- Encoder --------
        X_flat = X.reshape(B, L, N * F)         # (B, L, N*F)
        X_emb = self.enc_in(X_flat)             # (B, L, hidden)
        enc_out, (h, c) = self.encoder(X_emb)   # h/c: (num_layers, B, hidden)

        # Use last layer state to init decoder
        h_t = h[-1]  # (B, hidden)
        c_t = c[-1]

        h_t = self.norm(h_t)

        # Initial decoder input: last observed power at anchor time (from X last step feature 0)
        # X[:, -1, :, 0] is normalized anchor power P3
        y_prev = X[:, -1, :, 0]  # (B, N)

        preds = []
        for t in range(self.H):
            h_t, c_t = self.decoder_cell(y_prev, (h_t, c_t))
            h_t = self.norm(h_t)
            y_t = self.out_proj(h_t)  # (B, N)
            preds.append(y_t)

            if (teacher_forcing_y is not None) and (teacher_forcing_ratio > 0.0):
                use_tf = (torch.rand(B, device=X.device) < teacher_forcing_ratio).float().unsqueeze(-1)
                y_prev = use_tf * teacher_forcing_y[:, t, :] + (1.0 - use_tf) * y_t
            else:
                y_prev = y_t

        Y_hat = torch.stack(preds, dim=1)  # (B, H, N)
        return Y_hat
