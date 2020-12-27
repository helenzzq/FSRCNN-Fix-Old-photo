import torch
import torch.nn as nn


class FSRCNNModel(torch.nn.Module):
    def __init__(self, channel_num, upscale_factor, d=64, s=16, m=3):
        super(FSRCNNModel, self).__init__()
        # Set the argument for first 3 convolution layers
        convolution_augment = [(channel_num, d, 5, 1, 2), (d, s, 1, 1, 0), (s, d, 1, 1, 0)]
        modules = []
        for conv in convolution_augment:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels=conv[0], out_channels=conv[1], kernel_size=conv[2], stride=conv[3],
                          padding=conv[4]),
                nn.PReLU()))
        # Assign feature extraction layer
        self.feature_extraction = modules[0]

        # do the shrinking, mapping, and expanding
        # reduce the LR feature dimension d to s by 1*1 kernel
        self.shrink = [modules[1]]
        self.mapping_expanding(s, m, modules[2])

        # deconvolution by 9*9 kernel to rebuild the RH images
        self.expand = torch.nn.Sequential(*self.shrink)

        # Deconvolution
        self.deconvolution = nn.ConvTranspose2d(in_channels=d, out_channels=channel_num, kernel_size=9,
                                                stride=upscale_factor, padding=3, output_padding=1)

    def forward(self, x):
        """ By the FSRCNN, forward is used to apply to all three main layers"""
        temp = self.expand(self.feature_extraction(x))
        return self.deconvolution(temp)

    def mapping_expanding(self, s, ranges, modules):
        """Do the mapping according to given m,
            expand s back to the HR feature dimension d by 1*1 kernel"""
        for x in range(ranges):
            self.shrink.extend([nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=1)])
        self.shrink.append(nn.PReLU())# bug may appear here
        self.shrink.append(modules)

    def weight_init(self, mean, std):
        ''' give the init for weight and bias'''

        # torch.nn.init.normal(tensor, mean, std)
        # for feature extraction layer
        for j in self.feature_extraction:
            if isinstance(j, nn.Conv2d):
                nn.init.normal_(j.weight, mean=0, std=0.0378)
                nn.init.constant_(j.bias, 0)

        # for shrinking and expanding layers
        for k in self.shrink:
            if isinstance(k, nn.Conv2d):
                nn.init.normal_(k.weight, mean=0, std=0.1179)
                nn.init.constant_(k.bias, 0)

        # for deconvolution layers
        nn.init.normal_(self.deconvolution.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.deconvolution.bias, 0)
