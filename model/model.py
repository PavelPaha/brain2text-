from torch import nn
import torch


class GRU(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 device, 
                 days_count, 
                 kernel_size=5, 
                 stride=5):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.days_count = days_count
        self.kernel_size = kernel_size
        self.stride = stride

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_size, hidden_size) * 0.1) for _ in range(self.days_count)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.hidden_size)) for _ in range(self.days_count)]
        )

        self.convs = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, stride=self.stride),
            # nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, stride=self.stride),
        )
        self.conv_norm = nn.LayerNorm(self.hidden_size)

        self.normalization = nn.ReLU()
        # self.in_fc = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU()
        # )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.device = device


    def forward(self, batch):
        day_indices = batch['day_idx'] # [b, 1]

        day_weights = torch.stack([self.day_weights[i] for i in day_indices], dim=0) # [b, hidden_size, hidden_size]
        day_biases = torch.cat([self.day_biases[i].squeeze(1) for i in day_indices], dim=0).unsqueeze(1) # [b, hidden_size]

        x = batch['neural_data'].to(self.device) # [b, seq_len, input_size]
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases # [b, seq_len, hidden_size]
        # x = self.normalization(x)

        # seq_len_after_conv = (x.size(1) - self.kernel_size) // self.stride

        # x: [b, seq_len, hidden_size]
        x = x.permute(0, 2, 1) # [b, hidden_size, seq_len]
        x = self.convs(x) # [b, seq_len_after_conv, hidden_size]
        x = x.permute(0, 2, 1) # [b, seq_len_after_conv, hidden_size]
        x = self.conv_norm(x)

        # print(x.size(1), batch['phonemes_ids'].size(1))
        
        logits, hidden_states = self.gru(x)
        out = self.out_fc(hidden_states)
        return logits, out