from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.in_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):

        logits, hidden_states = self.gru(x)
        out = self.out_fc(hidden_states)
        return logits, out