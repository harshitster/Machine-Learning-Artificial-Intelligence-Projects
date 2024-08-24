import torch
import argparse
import os

import losses
import model
import dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(config):
    DCE_Net = model.DCENet().cuda()
    DCE_Net.apply(weights_init)

    if config.load_pretrain == True:
        DCE_Net.load_state_dict(torch.load(config.pretrain_dir))

    train_dataset = dataloader.DataLoader(config.train_data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    SCLoss = losses.SpatialConsistencyLoss()
    ELoss = losses.ExposureLoss()
    CCLoss = losses.ColorConstancyLoss()
    ISLoss = losses.IlluminationSmoothnessLoss(1)

    optimizer = torch.optim.Adam(DCE_Net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_Net.train()

    for epoch in range(config.n_epochs):
        for iteration, image_lowlight in enumerate(train_loader):
            image_lowlight = image_lowlight.cuda()

            enhanced_image_t, enhanced_image, alpha = DCE_Net(image_lowlight)

            sc_loss = torch.mean(SCLoss(image_lowlight, enhanced_image))
            e_loss = 10 * torch.mean(ELoss(enhanced_image))
            cc_loss = 5 * torch.mean(CCLoss(enhanced_image))
            is_loss = 200 * ISLoss(alpha)

            loss = sc_loss + e_loss + cc_loss + is_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_Net.parameters(), config.grad_clip)
            optimizer.step()

            if((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                torch.save(DCE_Net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_path', type=str, default="/Applications/ML projects/Success/Low Light Image Enhancement/lol_dataset/our485/low/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_dir', type=str, default= "")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)