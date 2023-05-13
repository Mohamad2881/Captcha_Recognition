import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    def __init__(self, max_num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(1152, 64)
        self.dropout_1 = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 32, 2, dropout=0.2, bidirectional=True, batch_first=True)
        self.out = nn.Linear(64, max_num_chars+1)  # 64 cuz bidirectional (32*2)

        self.max_num_chars = max_num_chars

    def forward(self, images, labels=None):
        # labels can be None in inference
        bs, c, h, w = images.size()
        # print(bs, c, h, w)

        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())

        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x)
        # print(x.size())  # --> [1, 64, 18, 75]

        x = x.permute(0, 3, 1, 2)  # --> [1, 75, 18, 64]
        x = x.view(bs, x.size(1), -1) # change this
        # print(x.size())  # --> [1, 75, 1152]

        x = F.relu(self.linear_1(x))
        x = self.dropout_1(x)
        # print(x.size())

        x, _ = self.gru(x)
        x = self.out(x)
        # print(x.size())

        x = x.permute(1, 0, 2)
        # print(x.size())

        if labels is not None:
            log_softmax_vals = F.log_softmax(x, dim=2)
            input_lengths = torch.full(size=(bs,), fill_value=log_softmax_vals.size(0), dtype=torch.int32)
            target_lengths = torch.full(size=(bs,), fill_value=labels.size(1), dtype=torch.int32)
            loss = nn.CTCLoss(blank=self.max_num_chars)(log_softmax_vals, labels, input_lengths, target_lengths)

            return x, loss

        return x, None

