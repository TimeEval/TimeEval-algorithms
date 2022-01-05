import torch.nn.functional as F
from torch.nn import Module, Conv1d, MaxPool1d, Linear, Dropout


class DeepAnTCNN(Module):
    def __init__(self, window, pred_window, in_channels, filter1_size, filter2_size, kernel_size, pool_size, stride):
        super(DeepAnTCNN, self).__init__()

        # layers
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=filter1_size, kernel_size=kernel_size, stride=stride, padding = 0)

        self.conv2 = Conv1d(in_channels=filter1_size, out_channels=filter2_size, kernel_size=kernel_size, stride=stride, padding = 0)

        self.maxpool = MaxPool1d(pool_size)

        self.dropout = Dropout(0.25)

        self.pred_window = pred_window
        self.in_channels = in_channels
        self.dim1 = int(0.5*(0.5*(window-1)-1)) * filter2_size
        self.lin1 = Linear(self.dim1, in_channels*pred_window)

    def forward(self, x):
        # convolution layer 1
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)

        # convolution layer 2
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, self.dim1)

        x = self.dropout(x)
        x = self.lin1(x)

        return x.view(-1, self.pred_window, self.in_channels)
