import torch

class SpatialConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        left_kernel = torch.tensor(
            [[0,0,0],[-1,1,0],[0,0,0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).cuda()

        right_kernel = torch.tensor(
            [[0,0,0],[0,1,-1],[0,0,0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).cuda()

        up_kernel = torch.tensor(
            [[0,-1,0],[0,1, 0 ],[0,0,0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).cuda()

        down_kernel = torch.tensor(
            [[0,0,0],[0,1, 0],[0,-1,0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).cuda()

        self.left_weight = torch.nn.Parameter(data=left_kernel, requires_grad=False)
        self.right_weight = torch.nn.Parameter(data=right_kernel, requires_grad=False)
        self.up_weight = torch.nn.Parameter(data=up_kernel, requires_grad=False)
        self.down_weight = torch.nn.Parameter(data=down_kernel, requires_grad=False)

        self.avg_pool = torch.nn.AvgPool2d(kernel_size=4)

    def forward(self, original, enhanced):
        original_mean = torch.mean(original, dim=1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)

        original_pool = self.avg_pool(original_mean)
        enhanced_pool = self.avg_pool(enhanced_mean)

        d_original_left = torch.nn.functional.conv2d(original_pool, weight=self.left_weight, padding=1)
        d_original_right = torch.nn.functional.conv2d(original_pool, weight=self.right_weight, padding=1)
        d_original_up = torch.nn.functional.conv2d(original_pool, weight=self.up_weight, padding=1)
        d_original_down = torch.nn.functional.conv2d(original_pool, weight=self.down_weight, padding=1)

        d_enhanced_left = torch.nn.functional.conv2d(enhanced_pool, weight=self.left_weight, padding=1)
        d_enhanced_right = torch.nn.functional.conv2d(enhanced_pool, weight=self.right_weight, padding=1)
        d_enhanced_up = torch.nn.functional.conv2d(enhanced_pool, weight=self.up_weight, padding=1)
        d_enhanced_down = torch.nn.functional.conv2d(enhanced_pool, weight=self.down_weight, padding=1)

        d_left = torch.pow(d_original_left - d_enhanced_left, 2)
        d_right = torch.pow(d_original_right - d_enhanced_right, 2)
        d_up = torch.pow(d_original_up - d_enhanced_up, 2)
        d_down = torch.pow(d_original_down - d_enhanced_down, 2)

        loss = d_left + d_right + d_up + d_down

        return loss
    

class ExposureLoss(torch.nn.Module):
    def __init__(self, kernel_size=16, mean_val=0.6):
        super().__init__()

        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size)
        self.mean_val = mean_val

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.pool(x)

        loss = torch.mean(torch.pow(x - torch.FloatTensor([self.mean_val]).cuda(), 2))
        
        return loss
    
class ColorConstancyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, dim=(2, 3), keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, split_size_or_sections=1, dim=1)

        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)

        loss = torch.sqrt(d_rg + d_gb + d_rb)
        loss = torch.mean(loss)

        return loss
    
class IlluminationSmoothnessLoss(torch.nn.Module):
    def __init__(self, coeff):
        super().__init__()

        self.coeff = coeff

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)

        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x - 1,:]), 2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x - 1]), 2).sum()

        loss = self.coeff * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return loss