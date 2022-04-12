# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import torch
from utils import load_adj_neg, load_dataset_adj_lap
from ssgc import Net
import argparse
import numpy as np
from classification import classify

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='citeseer',
                    help='dataset')
parser.add_argument('--seed', type=int, default=123,
                    help='seed')
parser.add_argument('--nhid', type=int, default=512,
                    help='hidden size')
parser.add_argument('--output', type=int, default=512,
                    help='output size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=80,
                    help='maximum number of epochs')
parser.add_argument('--sample', type=int, default=5,
                    help='    ')
parser.add_argument('--num_nodes', type=int, default=3327,
                    help='    ')
parser.add_argument('--num_features', type=int, default=3703,
                    help='    ')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature, adj_normalized, lap_normalized= load_dataset_adj_lap(args.dataset)
feature = feature.to(device)
adj_normalized = adj_normalized.to(device)
lap_normalized = lap_normalized.to(device)
K = 8
emb = feature#.clone()
for i in range(K):
    feature = torch.mm(adj_normalized, feature)
    emb = emb + feature
emb/=K
neg_sample= torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)

model = Net(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.train()
Lambda = 0.8
for epoch in range(args.epochs):

    optimizer.zero_grad()
    out = model(emb)
    loss = ((Lambda*torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out)) - torch.trace(
        torch.mm(torch.mm(torch.transpose(out, 0, 1), lap_normalized), out)))/lap_normalized.shape[0])
    print(loss)
    loss.backward()
    optimizer.step()

emb = model(emb).cpu().detach().numpy()
np.save('embedding.npy', emb)
classify(emb, args.dataset, per_class='20')
classify(emb, args.dataset, per_class='5')