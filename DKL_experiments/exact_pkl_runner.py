import bayesian_benchmarks as bb
from bayesian_benchmarks.data import regression_datasets, get_regression_data
import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import seaborn as sns
import argparse
import pandas as pd  # Add pandas import for DataFrame handling

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.models import SingleTaskVariationalGP

# Existing helper functions (get_mll, ConditionalMLL, RMSE, LargeFeatureExtractor, GPRegressionModel) go here...

def main(args):
    data = get_regression_data(args.dataset)
    train_x = torch.FloatTensor(data.X_train)[:args.ntrain]
    train_y = torch.FloatTensor(data.Y_train).squeeze()[:args.ntrain]
    test_x = torch.FloatTensor(data.X_test)
    test_y = torch.FloatTensor(data.Y_test).squeeze()
    data_dim = train_x.size(-1)
    m = int(args.m * train_x.shape[0])

    # Create an empty list to store the results for all trials
    results = []

    for trl in range(args.ntrial):
        feature_extractor = LargeFeatureExtractor(data_dim)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, feature_extractor)

        if torch.cuda.is_available():
            use_cuda = True
            model = model.cuda()
            likelihood = likelihood.cuda()
            train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

        training_iterations = 100
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters()},
            {'params': model.covar_module.parameters()},
            {'params': model.mean_module.parameters()},
            {'params': model.likelihood.parameters()},
        ], lr=0.01)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        def train(losstype='cmll'):
            iterator = tqdm.tqdm(range(training_iterations))
            for i in iterator:
                optimizer.zero_grad()
                output = model(train_x)
                if losstype == 'mll':
                    loss = -mll(output, train_y)
                if losstype == 'cmll':
                    order = torch.randperm(train_x.shape[0])
                    xm = train_x[order[:m]]
                    ym = train_y[order[:m]]
                    loss = -ConditionalMLL(model, train_x, train_y, xm, ym)
                loss.backward()
                iterator.set_postfix(loss=loss.item())
                optimizer.step()

        train(losstype=args.losstype)
        model.eval()
        test_preds = model(test_x).mean
        rmse = RMSE(test_preds, test_y)

        # Append the result to the results list
        results.append({
            'Dataset': args.dataset,
            'Type': args.losstype,
            'm': args.m,
            'N': args.ntrain,
            'RMSE': rmse.item()  # Convert the tensor to a scalar
        })

    # Convert results into a DataFrame
    df = pd.DataFrame(results)

    # Save the DataFrame to a .pkl file
    fpath = "./saved-outputs/"
    fname = f"exact_uci_df_{args.dataset}_ntrain{args.ntrain}_m{args.m}_{args.losstype}.pkl"
    df.to_pickle(fpath + fname)
    print(f"Results saved to {fpath + fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=0.9)
    parser.add_argument('--losstype', type=str, default='mll')
    parser.add_argument('--dataset', type=str, default='mll')
    parser.add_argument('--ntrain', type=int, default=200)
    parser.add_argument('--ntrial', type=int, default=10)
    args = parser.parse_args()

    main(args)
