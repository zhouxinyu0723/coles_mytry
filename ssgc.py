import torch


class Net(torch.nn.Module):
    def __init__(self, args,     bias = True):
        super(Net, self).__init__()


        self.conv1 = torch.nn.Linear(args.num_features, args.output, bias = bias)


    def forward(self, F1):

        z = self.conv1(F1)


        return z

class MLP(torch.nn.Module):
    def __init__(self, args, nclass,    bias = True):
        super(MLP, self).__init__()


        self.conv1 = torch.nn.Linear(args.num_features, args.num_features, bias = bias)
        self.conv2 = torch.nn.Linear(args.num_features, nclass, bias = bias)


    def forward(self, z):

        z = self.conv1(z)
        z = self.conv2(z)
        

        return z

