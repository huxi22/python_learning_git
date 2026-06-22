import torch
import torch.nn as nn

class UWB_CRNN(nn.Module):
    def __init__(self, cir_len = 640, num_bins = 200):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d( 1, 16, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2)
        )

        self.projection = nn.Sequential(
            nn.Linear(32*(cir_len//4), 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_bins), nn.Sigmoid()
        )


    def forward(self, x):
        batch, seq, cir = x.size()
        x = x.view(batch*seq, 1, cir)

        feat = self.cnn(x)
        feat = feat.view(feat.size(0), -1)

        feat = self.projection(feat)
        lstm_in = feat.view(batch, seq, -1)

        lstm_out, _ = self.lstm(lstm_in)
        out  = self.decoder(lstm_out)

        return out



model = UWB_CRNN()

x = torch.rand(20, 30, 640)
y = model(x)

print(x.shape)
print(y.shape)














