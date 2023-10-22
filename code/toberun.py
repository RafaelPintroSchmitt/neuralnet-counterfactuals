import os
os.chdir(r"C:\Users\rafap\Desktop\RA2023 - personal files\neuralnet-counterfactuals")

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import torch_geometric.utils
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch_geometric.nn import GATConv # [^1^][1]

# Load the dta datasets into a pandas dataframe
df2006 = pd.read_stata(rf"data\brazilian_municipalities2006.dta")
df2010 = pd.read_stata(rf"data\brazilian_municipalities2010.dta")
df2014 = pd.read_stata(rf"data\brazilian_municipalities2014.dta")
df2018 = pd.read_stata(rf"data\brazilian_municipalities2018.dta")
df2022 = pd.read_stata(rf"data\brazilian_municipalities2022.dta")

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
device_cpu = torch.device('cpu')

# %% [markdown]
# Now, I create a list containing the graphs of each election. Each graph is a tensor_geometric Data object, which contains the adjacency matrix and the features of each node. The graph structure is defined by linking each city to it's 10 closest peers (in terms of geographical distance).
# 
# Roughly, I take the datasets for the years 2010 through 2022 and I define the output vector (Y) as the difference in vote-share in the runoff for the PT (workers party) between the current and last election. The X matrix, containing covariates, includes city-level covariates extracted from the Brazilian 2010 census. Note that it also includes the output vector (difference in vote-share). The goal here is to train the network by (1) dropping some cities in the training set arbitralily, and (2) predicting the vote-share of the PT in those cities through convolutions/attention layers. That is, I leverage "local" voting patterns to extrapolate voting behavior to the omitted-outcome cities. The network should thus be learning an underlying geographical contagion/distribution process of voting behavior. In evaluating the model, I will only use cities out of the training set. In particular, I will drop "diff" for half the cities in the test set and predict their vote-share based on the other half (see the definition of Z).
# 
# I also define treatment variables here - i.e. I drop the "labels" or outcomes of the treated cities, and will use the non-dropped controls to predict the counterfactuals for the treated cities.

# %%
data_list = []
treatment_list = []
treatment_mask = []
Z_list = []
Zmask_list = []
#loop over years
for df in [df2010, df2014, df2018, df2022]:
    # encode the cateogrical variables as integers
    df["regiao"] = df["regiao"].astype("category").cat.codes
    df["sigla_uf"] = df["sigla_uf"].astype("category").cat.codes

    #drop missing values
    df = df[~np.isnan(df).any(axis=1)]

    #create difference from previous election to current election
    df["diff"] = df["pt_share"] - df["l4pt_share"]
    Y = df.loc[:, ["diff"]]
    #divide Y in 100 quantiles
    Y = pd.qcut(Y["diff"], q=100, labels=False)
    Y = Y.to_numpy()

    X = df.loc[:, ["diff", "l4pt_share", "populacao", "evangelico", "urban", "radio", "televisao", "idade", "alfabetizado", "rend_total", "area", "density", "white", "nasceu_mun", "horas_trabprin", "filhos_nasc_vivos", "high_school", "bachelor", "vive_conjuge", "sexo", "high"]]
    #save the collumn high in a pd series
    high = X["high"]
    #normalize the data
    X = (X - X.mean()) / X.std()
    #add the collumn high to the normalized data
    X["high"] = high

    # Create treat, which is a copy of df with the values of diff set to 0 if high (my treatment variable) is 1
    # This is to evaluate the effect of treatment in 2022
    treat = X.copy().to_numpy()
    treat[:, 0] = treat[:, 0] * (1 - treat[:, 20])
    #drop collumn high from treat
    treat = treat[:, :-1]
    # Create a tensor for treat
    treat = torch.tensor(treat, dtype=torch.float).to(device)
    # Create a boolean taking values 1 if treat is 0 and 1 otherwise (use sourceTensor.clone().detach())
    treat_mask = treat[:, 0].clone().detach()
    treat_mask[treat_mask != 0] = 1
    treat_mask = torch.logical_not(treat_mask).to(device)
    treatment_list.append(treat)
    treatment_mask.append(treat_mask)

    #drop the collumn high
    X = X.drop(columns=["high"])

    # Create Z, which is a copy of X with a random 50% of the values in diff set to 0
    # This is for testing the model at the end
    Z = X.copy().to_numpy()
    Z[:, 0] = np.random.choice([0, 1], size=Z.shape[0], p=[0.5, 0.5]) * Z[:, 0]
    # Create a tensor for Z
    Z = torch.tensor(Z, dtype=torch.float).to(device)
    # Create a boolean taking values 1 if Z is 0 and 1 otherwise (use sourceTensor.clone().detach())
    Z_mask = Z[:, 0].clone().detach()
    Z_mask[Z_mask != 0] = 1
    Z_mask = torch.logical_not(Z_mask).to(device)
    Z_list.append(Z)
    Zmask_list.append(Z_mask)

    # Extract the coordinates as a numpy array
    coords = df[["_Y0", "_X0"]].to_numpy()

    # Alternatively, use a k-nearest neighbors algorithm to find the 50 nearest neighbors for each city
    nn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    nn.fit(coords)
    dist_sparse = nn.kneighbors_graph(mode="distance")

    # Convert the sparse matrix to edge index and edge weight tensors
    edge_index, edge_attr = torch_geometric.utils.convert.from_scipy_sparse_matrix(dist_sparse)

    # Make X and Y tensors
    X = torch.tensor(X.to_numpy(), dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    data = Data(x=X, y=Y, edge_attr=edge_attr, edge_index=edge_index)

    # Get the number of cities
    num_cities = data.num_nodes

    # Randomly select 2500 cities for training
    perm = torch.randperm(num_cities)
    train_idx = perm[:2500]

    # Split the remaining cities into validation and test sets
    val_idx, test_idx = torch.split(perm[2500:], num_cities // 2)

    # Create boolean masks for each set
    train_mask = torch.zeros(num_cities, dtype=torch.bool)
    train_mask = train_mask.scatter(0, train_idx, 1)

    val_mask = torch.zeros(num_cities, dtype=torch.bool)
    val_mask = val_mask.scatter(0, val_idx, 1)

    test_mask = torch.zeros(num_cities, dtype=torch.bool)
    test_mask = test_mask.scatter(0, test_idx, 1)

    # Add the masks as attributes to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Convert the labels to long tensors
    data.y = data.y.long()

    data.validate(raise_on_error=True)

    in_features = X.shape[1]
    #define out_features as the number of quantiles in Y, as a function of Y
    out_features = len(np.unique(Y))

    #put in gpu
    data.to(device)

    #append to data_list
    data_list.append(data)


# %% [markdown]
# To test whether the predictions also generalize across years, I take 2022 (and possibly 2018) away from the training set. I'll then apply the model to 2022 and check the performance.

# %%
# store all the elections
data2010 = data_list[0]
data2014 = data_list[1]
data2018 = data_list[2]

# store the last item of data_list to data2022
data2022 = data_list[-1]
data_list.pop(-1)
#data_list.pop(-1) #if want to train without 2018
data_list

# %% [markdown]
# I now turn to defining the model. I will use a Graph Attention Network (GAT). I have a dropout layer, which omits the information of some nodes in each layer. I then have two GAT layers, with a RELU activation function in between, which are the core of the model. Finally, I have a linear layer, which outputs the predicted vote-share for the PT in each city. The model is defined as follows:

# %%
import torch.nn as nn
import random
j = random.randint(0, 1000)
class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(j)
        self.conv1 = GATConv(in_features, in_features, heads=2)
        self.conv2 = GATConv(in_features*2, in_features, heads=2)
        self.lin1 = Linear(in_features*2, out_features)

    def forward(self, x, edge_index, edge_attr):
        #make drops start at 0.9 and decrease by 0.05 every 10 epochs, until it reaches 0.5
        drops = 0.9 - (0.05 * (epoch // 10))
        if drops < 0.5:
            drops = 0.5
        x = F.dropout(x, p=drops, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.lin1(x)
        return x

model = GAT()
model = model.to(device)

# %% [markdown]
# The training of the model is simple: for every epoch I actually take one optimization step for each election included in the training. By alternating the elections at every training step (instead of training on each election sequentially), I hope to capture a more robust relationship underlying electoral geographical distributions. Note that I start by dropping 90% of the nodes in the dropout layer, going down to 50% as the training progresses.

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss().to(device)

def train():
    #loop over items of datalist
    for data in data_list:
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.edge_attr)  # Perform a single forward pass.
        # Compute the loss solely based on the training nodes.
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return loss

for epoch in range(1, 100):
    drops = 0.9
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# %% [markdown]
# Testing is defined as follows. There are 3 functions. The main one (test(year)), is used to test the model's predictions for the treatment groups defined above. Unexpected biases in the predictions of the model are interpreted as the effect of treatment (since we are comparing with the models "counterfactual" predictions).
# 
# The other two are simple test functions, which evaluate the models capabilities to predict vote-shares of a sample of randomly selected cities given the remaining ones' vote-shares in 2022. More precisely: Z is defined above by taking data2022 and setting half of the values of diff to 0. Z_mask is a boolean vector taking value 1 if diff was set to 0 in a particular observation. Here, I evaluate the models ability to predict diff for those cities that had their diff set to 0.
# 
# test_testset() evaluates the model's fit for the test set, whereas test_trainset() evaluates the model's fit for the training set (the latter is mainly to diagnose overfitting in case the accuracy in the training set is much larger than in the test set.)

# %%
def test(year):
      model.eval()
      #test on each dataset in datalist
      data = data_list
      #data.append(data2018) #if trained without 2018
      data.append(data2022)
      data = data[int((year+2)/4-503)] #the weird integer is just the list index (0,1,2,3) from the year list (2010, 2014, 2018, 2022)
      out = model(treatment_list[int((year+2)/4-503)], data.edge_index, data.edge_attr)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      #note: no need to use treatment_mask list since the selection across years is constant (what changes is the dataset with diff)
      mask = treat_mask*data.test_mask #add *data.test_mask if year is in in the training set (so we exclude the training cities for each year)
      test_correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      # calculate the mean squared error, make input a float tensor
      mse = torch.mean((pred[mask].float() - data.y[mask].float())**2)
      # bias
      bias = torch.mean((pred[mask].float() - data.y[mask].float()))
      # generate a tensor with the values of data.y[mask] but randomly shuffled
      shuffled = data.y[mask].float().clone()
      shuffled = shuffled[torch.randperm(len(shuffled))]
      #mse with suffled data
      mse_shuffled = torch.mean((shuffled - data.y[mask].float())**2)
      #fake rsquared
      r2 = 1 - mse/mse_shuffled
      #plot the two histograms together, with transparency and different colors
      plt.hist(pred[mask].cpu().float() - data.y[mask].cpu().float(), alpha=0.5, label='Predictions')
      plt.hist(shuffled.cpu() - data.y[mask].cpu().float(), alpha=0.5, label='Shuffled')
      plt.legend(loc='upper right')
      plt.show()
      return test_acc, pred, mse, bias, r2, out

#Test on the test data
def test_testset(year):
      model.eval()
      #test on each dataset in datalist
      data = data_list
      #data.append(data2018) #if trained without 2018
      data.append(data2022)
      data = data[int((year+2)/4-503)] #the weird integer is just the list index (0,1,2,3) from the year list (2010, 2014, 2018, 2022)
      out = model(Z_list[int((year+2)/4-503)], data.edge_index, data.edge_attr)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      mask = Zmask_list[int((year+2)/4-503)]*data.test_mask #add *data.test_mask if year is in in the training set (so we exclude the training cities for each year)
      test_correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      # calculate the mean squared error, make input a float tensor
      mse = torch.mean((pred[mask].float() - data.y[mask].float())**2)
      # bias
      bias = torch.mean((pred[mask].float() - data.y[mask].float()))
      # generate a tensor with the values of data.y[mask] but randomly shuffled
      shuffled = data.y[mask].float().clone()
      shuffled = shuffled[torch.randperm(len(shuffled))]
      #mse with suffled data
      mse_shuffled = torch.mean((shuffled - data.y[mask].float())**2)
      #fake rsquared
      r2 = 1 - mse/mse_shuffled

      #plot the two histograms together, with transparency and different colors
      plt.hist(pred[mask].cpu().float() - data.y[mask].cpu().float(), alpha=0.5, label='Predictions')
      plt.hist(shuffled.cpu() - data.y[mask].cpu().float(), alpha=0.5, label='Shuffled')
      plt.legend(loc='upper right')
      #save plot
      plt.savefig(rf"outputs\performance.pdf")
      return test_acc, pred, mse, bias, r2, out

#test on the training data
def test_trainset():
        model.eval()
        data = data2022
        out = model(Z, data.edge_index, data.edge_attr)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        mask = Z_mask*data.train_mask #add *data.train_mask if data2022 in the training set
        test_correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
        return test_acc

