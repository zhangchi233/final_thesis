from torch import nn
import math
import torch
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model=1024, n_heads=4, d_keys=1024, d_llm=1024, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)


    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        B,S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(B,S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(B,S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)
        
        return self.out_projection(out)
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,bshe->blhe", A, value_embedding)

        return reprogramming_embedding
if __name__ == '__main__':
    repro = ReprogrammingLayer()
    list(repro.parameters())