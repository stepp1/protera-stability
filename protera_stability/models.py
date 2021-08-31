from torch import nn

class ProteinMLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_units: int,
        n_layers: int,
        act: nn.Module,
        drop_p: int,
        last_drop: bool,
    ):
        super(ProteinMLP, self).__init__()

        layers = []
        for i in range(n_layers):
            in_feats = n_in if i == 0 else n_units // i
            if i == 0:
                in_feats = n_in
                out_feats = n_units
            else:
                in_feats = out_feats
                out_feats = in_feats // 2

            out_feats = 1 if i == (n_layers - 1) else out_feats
            fc = nn.Linear(in_feats, out_feats)
            layers.append(fc)
        self.layers = nn.ModuleList(layers)

        self.drop = nn.Dropout(p=drop_p)
        self.last_drop = last_drop
        self.act = act
        if act is None:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == len(self.layers) - 1:
                continue
            else:
                out = self.act(self.drop(out))

        if self.last_drop:
            out = self.drop(out)
        return out
