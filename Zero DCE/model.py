import torch

class DCENet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        feature_maps = 32

        self.conv1 = torch.nn.Conv2d(3, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = torch.nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = torch.nn.Conv2d(feature_maps, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = torch.nn.Conv2d(feature_maps * 2, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = torch.nn.Conv2d(feature_maps * 2, feature_maps, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = torch.nn.Conv2d(feature_maps * 2, 24, kernel_size=3, stride=1, padding=1, bias=True)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))

        x5 = self.relu(self.conv5(torch.cat((x3, x4), dim=1)))
        x6 = self.relu(self.conv6(torch.cat((x2, x5), dim=1)))
        x_r = torch.tanh(self.conv7(torch.cat((x1, x6), dim=1)))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, split_size_or_sections=3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_t = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_t + r5 * (torch.pow(enhance_image_t, 2) - enhance_image_t)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        r = torch.cat((r1, r2, r3, r4, r5, r6, r7, r8), dim=1)

        return enhance_image_t, enhance_image, r