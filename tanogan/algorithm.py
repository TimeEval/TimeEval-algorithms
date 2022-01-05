import argparse
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import sys
import torch
from torch.autograd import Variable
from torch.nn import BCELoss
from torch.nn.init import normal_ as normal
from torch.optim import Adam
from torch.utils.data import DataLoader

from tanogan.dataset import TAnoGANDataset
from tanogan.models import LSTMGenerator, LSTMDiscriminator
from tanogan.early_stopping import EarlyStopping


@dataclass
class CustomParameters:
    epochs: int = 1
    cuda: bool = False
    window_size: int = 30
    learning_rate: float = 2e-4
    batch_size: int = 32
    n_jobs: int = 1
    iterations: int = 25
    random_state: int = 42
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10


class AlgorithmArgs(argparse.Namespace):
    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def train(args: AlgorithmArgs):
    data = args.df

    split_at = int(len(data) * args.customParameters.split)

    dataset = TAnoGANDataset(X=data.values[:split_at, 1:-1], y=data.values[:split_at, -1],
                             window_length=args.customParameters.window_size,
                             stride=1)

    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=args.customParameters.batch_size,
                            num_workers=args.customParameters.n_jobs)

    valid_dataset = TAnoGANDataset(X=data.values[split_at:, 1:-1], y=data.values[:split_at, -1],
                             window_length=args.customParameters.window_size,
                             stride=1)

    valid_dataloader = DataLoader(valid_dataset, shuffle=True,
                            batch_size=args.customParameters.batch_size,
                            num_workers=args.customParameters.n_jobs)

    device = torch.device("cuda:0" if args.customParameters.cuda else "cpu")  # select the device
    in_dim = dataset.n_feature  # input dimension is same as number of feature

    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)

    criterion = BCELoss().to(device)
    optimizerG = Adam(netG.parameters(), lr=args.customParameters.learning_rate)
    optimizerD = Adam(netD.parameters(), lr=args.customParameters.learning_rate)

    real_label = 1
    fake_label = 0

    def save_model():
        torch.save({
            "discriminator": netD.state_dict(),
            "generator": netG.state_dict(),
            "in_dim": in_dim
        }, args.modelOutput)

    early_stopping = EarlyStopping(args.customParameters.early_stopping_patience, args.customParameters.early_stopping_delta, args.customParameters.epochs,
                                   callbacks=[(lambda i, _l, _e: save_model() if i else None)])

    for epoch in early_stopping:
        netD.train()
        for i, (x, y) in enumerate(dataloader, 0):
            # Train with real data
            netD.zero_grad()
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device, dtype=torch.float)

            output, _ = netD.forward(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            optimizerD.step()
            D_x = output.mean().item()

            # Train with fake data
            noise = Variable(normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1))
            if args.customParameters.cuda:
                noise = noise.cuda()
            fake, _ = netG.forward(noise)
            output, _ = netD.forward(fake.detach())
            label.fill_(fake_label)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train G
            netG.zero_grad()
            noise = Variable(normal(torch.Tensor(batch_size, seq_len, in_dim), mean=0, std=0.1))
            if args.customParameters.cuda:
                noise = noise.cuda()
            fake, _ = netG.forward(noise)
            label.fill_(real_label)
            output, _ = netD.forward(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            D_G_z2 = output.mean().item()

        netD.eval()
        val_loss = []
        for i, (x, y) in enumerate(valid_dataloader, 0):
            real = x.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device, dtype=torch.float)

            output, _ = netD.forward(real)
            errD_real = criterion(output, label)
            val_loss.append(errD_real.item())
        early_stopping.update(np.mean(val_loss).item())


        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.customParameters.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    save_model()


def execute(args: AlgorithmArgs):
    data = args.df

    device = torch.device("cuda:0" if args.customParameters.cuda else "cpu")

    checkpoint = torch.load(args.modelInput)
    in_dim = checkpoint["in_dim"]

    netD = LSTMDiscriminator(in_dim=in_dim, device=device).to(device)
    netD.load_state_dict(checkpoint["discriminator"])

    netG = LSTMGenerator(in_dim=in_dim, out_dim=in_dim, device=device).to(device)
    netG.load_state_dict(checkpoint["generator"])

    def anomaly_score(x: Variable, G_z: torch.Tensor, Lambda=0.1):
        residual_losses = torch.abs(x - G_z).sum(axis=[1,2])
        output, x_feature = netD(x.to(device))
        output, G_z_feature = netD(G_z.to(device))

        discrimination_losses = torch.abs(x_feature - G_z_feature).sum(axis=[1,2])
        single_losses = (1-Lambda) * residual_losses.to(device) + Lambda * discrimination_losses.to(device)
        total_loss = (1-Lambda) * residual_losses.sum().to(device) + Lambda * discrimination_losses.sum().to(device)
        return total_loss, single_losses.detach().numpy()

    dataset = TAnoGANDataset(X=data.values[:, 1:-1], y=data.values[:, -1],
                             window_length=args.customParameters.window_size,
                             stride=args.customParameters.window_size)

    dataloader = DataLoader(dataset, shuffle=False,
                            batch_size=args.customParameters.batch_size,
                            num_workers=args.customParameters.n_jobs)

    loss_list = []
    for i, (x, y) in enumerate(dataloader):
        z = Variable(normal(torch.zeros(x.shape),
                            mean=0,
                            std=0.1),
                     requires_grad=True)
        z_optimizer = Adam([z], lr=1e-2)

        single_losses = None
        if args.customParameters.cuda:
            for j in range(args.customParameters.iterations):
                gen_fake, _ = netG(z.cuda())
                loss, single_losses = anomaly_score(Variable(x).cuda(), gen_fake)
                loss.backward()
                z_optimizer.step()
        else:
            for j in range(args.customParameters.iterations):
                gen_fake, _ = netG(z)
                loss, single_losses = anomaly_score(Variable(x), gen_fake)
                loss.backward()
                z_optimizer.step()
        print(single_losses.shape)
        loss_list.append(single_losses)
    loss_list = np.concatenate(loss_list)
    anomaly_scores = np.array([loss / args.customParameters.window_size for loss in loss_list])
    anomaly_scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
