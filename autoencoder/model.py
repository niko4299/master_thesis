import torch.nn as nn


class CAEC(nn.Module):

    def __init__(self,in_channels,encoder_arhitecture_layers_size,decoder_arhitecture_layers_size):
        super(CAEC, self).__init__()
        self.encoder = self.build_encoder(in_channels,encoder_arhitecture_layers_size)
        self.decoder = self.build_decoder(decoder_arhitecture_layers_size)

    def forward(self, x):
        x = self.encoder_forward(x)

        return self.decoder_forward(x)
    
    def encoder_forward(self, x):

        return self.encoder(x)
    
    def decoder_forward(self, x):
        return self.decoder(x)
    
    def build_encoder(self,in_channels,arhitecture_layers_size):
        layers = []
        for layer_size in arhitecture_layers_size:
            if layer_size == 'max':
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
            else:
                layers += [nn.Conv2d(in_channels = in_channels,out_channels=layer_size,kernel_size=(3,3),stride=(1,1),padding=(1,1)),nn.BatchNorm2d(layer_size),nn.ReLU()]
                in_channels = layer_size
        
        layers += [nn.AvgPool2d(kernel_size = (14,14), stride = (1,1))]
        layers += [Squeeze(dims = [-1], n_squeezes=[2])]
        
        return nn.Sequential(*layers)
    
    def build_decoder(self,decoder_arhitecture_layers_size):
        layers = []
        layers +=[nn.Linear(512,decoder_arhitecture_layers_size[0])]
        layers +=[Unsqueeze(dims = [-1],n_unsqueezes=[2])]
        layers += [nn.ConvTranspose2d(decoder_arhitecture_layers_size[0],decoder_arhitecture_layers_size[1],kernel_size=(7,7),stride = (2,2)),nn.ReLU()]
        in_channels = decoder_arhitecture_layers_size[1]
        for layer_size in decoder_arhitecture_layers_size[2:]:
            layers += [nn.ConvTranspose2d(in_channels,layer_size,kernel_size=(2,2),stride = (2,2)),nn.ReLU()]
            in_channels = layer_size
        
        return nn.Sequential(*layers)

class Unsqueeze(nn.Module):

    def __init__(self, dims, n_unsqueezes):
        super(Unsqueeze, self).__init__()
        self.n_unsqueezes = n_unsqueezes
        self.dims = dims

    def forward(self,x):
        for dim,times in zip(self.dims,self.n_unsqueezes):
            for _ in range(times):
                x = x.unsqueeze(dim = dim)
        return x


class Squeeze(nn.Module):

    def __init__(self, dims, n_squeezes):
        super(Squeeze, self).__init__()
        self.n_squeezes = n_squeezes
        self.dims = dims

    def forward(self,x):
        for dim,times in zip(self.dims,self.n_squeezes):
            for _ in range(times):
                x = x.squeeze(dim = dim)
        return x