from torch import nn

class Generator(nn.Module):
    def __init__(self, ngpu) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
