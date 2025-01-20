import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from modules import Encoder, Decoder, VAE
from utils import MNISTDataset, BCELoss

import argparse
from collections import OrderedDict
import json

def main(annotations_file_train, img_dir_train,
         device, workers, learning_rate, batch_size, epochs,
         z_dim, hidden_dim):
    dataset_train = MNISTDataset(annotations_file_train, img_dir_train)
    loader_train = DataLoader(dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)
    
    x_dim = len(dataset_train[0][0])
    encoder = Encoder(x_dim, hidden_dim, z_dim)
    decoder = Decoder(z_dim, hidden_dim, x_dim)
    vae = VAE(encoder, decoder).to(device)
    optimiser = Adam(vae.parameters(), lr=learning_rate)

    print("Training VAE...")
    vae.train()
    i = 1
    curr_loss = None
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {i}:")
        for batch in tqdm(loader_train):
            optimiser.zero_grad()
            x = batch[0].to(device)
            x_hat, mean, log_var = vae(x)
            loss = BCELoss().loss(x, x_hat, mean, log_var)
            total_loss += loss.item()
            loss.backward()
            optimiser.step()
        print(f"Epoch {i} complete.\tAverage loss: {total_loss / batch_size}")
        i += 1
        curr_loss = total_loss
    print("Traning complete.")

    hyperparameters = OrderedDict([
        ("annotations_file_train", annotations_file_train),
        ("img_dir_train", img_dir_train),
        ("device", device),
        ("workers", workers),
        ("learning_rate", learning_rate),
        ("batch_size", batch_size),
        ("epochs", epochs),
        ("z_dim", z_dim),
        ("hidden_dim", hidden_dim)
    ])
    log = {
        "hyperparameters": hyperparameters,
        "loss": curr_loss
    }
    torch.save(vae.state_dict(), "vae-state-dict.pt")
    torch.save(optimiser.state_dict(), "optimiser-state-dict.pt")
    with open("log.json", 'w') as fp:
        json.dump(log, fp)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",
                        default="./data/train/images/", type=str, required=False,
                        help="Training data directory",
                        dest="img_dir_train")
    parser.add_argument("--train-labels",
                        default="./data/train/labels.csv", type=str, required=False,
                        help="Training labels csv file",
                        dest="annotations_file_train")
    parser.add_argument("-d", "--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        type=str, required=False,
                        help="Specify the selected device",
                        dest="device")
    parser.add_argument("-w", "--workers",
                        default=1, type=int, required=False,
                        help="Number of workers",
                        dest="workers")
    parser.add_argument("-l", "--learning-rate",
                        default=1E-4, type=float, required=False,
                        help="Optimisation learning rate",
                        dest="learning_rate")
    parser.add_argument("-b", "--batch-size",
                        default=100, type=int, required=False,
                        help="Number of samples per batch",
                        dest="batch_size")
    parser.add_argument("-e", "--epochs",
                        default=30, type=int, required=False,
                        help="Number of epochs",
                        dest="epochs")
    parser.add_argument("-z", "--z-dim",
                        default=64, type=int, required=False,
                        help="Dimension of latent space",
                        dest="z_dim")
    parser.add_argument("--hidden-dim",
                        default=256, type=int, required=False,
                        help="Dimension of hidden layers",
                        dest="hidden_dim")
    args = parser.parse_args()
    main(args.annotations_file_train, args.img_dir_train,
         args.device, args.workers, args.learning_rate, args.batch_size, args.epochs,
         args.z_dim, args.hidden_dim)