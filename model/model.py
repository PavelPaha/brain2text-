from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        self.in_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.in_fc(x)
        logits, hidden_states = self.gru(x)
        out = self.out_fc(hidden_states)
        return logits, out