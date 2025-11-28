import torch
import torch.nn as nn

class AdaptiveGraph(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.E  = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self):
        M1 = self.W1(self.E)
        M2 = self.W2(self.E)
        A = torch.relu(M1 @ M2.T)
        return torch.softmax(A, dim=-1)

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, X, A):
        X_g = torch.einsum("nm,bmf->bnf", A, X)
        return self.lin(X_g)

class AGCRNCell(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.gc_z = GraphConv(in_dim+hid_dim, hid_dim)
        self.gc_r = GraphConv(in_dim+hid_dim, hid_dim)
        self.gc_h = GraphConv(in_dim+hid_dim, hid_dim)

    def forward(self, x_t, h_prev, A):
        inp = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.gc_z(inp, A))
        r = torch.sigmoid(self.gc_r(inp, A))
        inp_r = torch.cat([x_t, r*h_prev], dim=-1)
        h_tilde = torch.tanh(self.gc_h(inp_r, A))
        return (1-z)*h_prev + z*h_tilde

class AGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, embed_dim, horizon):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.graph = AdaptiveGraph(num_nodes, embed_dim)
        self.cell  = AGCRNCell(input_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, horizon)

    def forward(self, X):
        B, L, N, F = X.shape
        A = self.graph()
        h = torch.zeros(B, N, self.hidden_dim, device=X.device)
        for t in range(L):
            h = self.cell(X[:,t], h, A)
        return self.head(h).permute(0,2,1)
