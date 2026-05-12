import abc
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from torch import nn
import torch
import torch.nn.functional as F
# Evaluators
class BaseEvaluator(abc.ABC):
    def __init__(self, y_mean: float = None, y_std: float = None, log_scale: bool = True):
        self.y_mean = y_mean
        self.y_std = y_std
        self.log_scale = log_scale

    def __call__(self, y_true, y_pred):
        if self.y_mean is not None and self.y_std is not None:
            y_true = y_true * self.y_std + self.y_mean
            y_pred = y_pred * self.y_std + self.y_mean
        if self.log_scale:
            y_true = np.exp(y_true)
            y_pred = np.exp(y_pred)
        return self.eval(y_true, y_pred)

    @abc.abstractmethod
    def eval(self, y_true, y_pred):
        """Evaluate and return the score."""


class RMSE(BaseEvaluator):
    def eval(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) ** 0.5


class MAPE(BaseEvaluator):
    def eval(self, y_true, y_pred):
        if y_true.min()< 1e-6:
            y_true[abs(y_true) < 1e-6] = np.inf
        diff = np.abs((y_true - y_pred) / y_true)
        return np.mean(diff) * 100


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

from sklearn.model_selection import train_test_split

def random_fit_and_eval_new(X, y, models, evaluators, train_frac: float= 0.7, seed: int = 0, need_normalize: bool = False, feature_name: list = None, return_prediction: bool = False, return_train_loss: bool = False):

    set_seed(seed)

    # Random train-test split
    N = int(train_frac * len(X))
    perm = np.random.permutation(len(X))
    X_train, X_test = X[perm[:N]], X[perm[N:]]
    y_train, y_test = y[perm[:N]], y[perm[N:]]
    

    return y_train, y_test

def random_fit_and_eval(X, y, models, evaluators, train_frac: float= 0.7, seed: int = 0, need_normalize: bool = False, feature_name: list = None, return_prediction: bool = False, return_train_loss: bool = False, train_idx=None, test_idx=None, top_k=20):
    set_seed(seed)

    if train_idx and test_idx:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Random train-test split
        N = int(train_frac * len(X))
        perm = np.random.permutation(len(X))
        X_train, X_test = X[perm[:N]], X[perm[N:]]
        y_train, y_test = y[perm[:N]], y[perm[N:]]

    # print(X_train.shape,  y_train.shape, X_test.shape, y_test.shape)
    # # add weight
    # # Assign weights: higher weights for larger y values
    # weights = np.where(y > 6, 2, 1)  # Example: weight of 2 for y > 400, else 1

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=(1-train_frac), random_state=seed)

    if need_normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    scores = []
    feature_importances = []
    for model in models:
        model.fit(X_train, y_train)

        # model.fit(X_train, y_train, sample_weight=weights_train)
        y_train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        if return_train_loss:
            scores.append([[evaluator(y_train, y_train_pred), evaluator(y_test, y_pred)] for evaluator in evaluators])
        else:
            scores.append([evaluator(y_test, y_pred) for evaluator in evaluators])

        if feature_name and len(feature_name)>0:
            importances = model.feature_importances_

            # Filter nonzero features only
            nonzero_idxs = np.where(importances > 0)[0]

            # Sort remaining features by importance
            sorted_nonzero_idxs = nonzero_idxs[np.argsort(importances[nonzero_idxs])[::-1]]

            # Select top-K
            top_idxs = sorted_nonzero_idxs[:top_k]

            # Pack results (feature name, importance)
            feature_importances = [(feature_name[i], importances[i]) for i in top_idxs]
            # fea_masks = model.feature_importances_ != 0
            # fea_importance = []
            # # Get feature importances  
            # importances = model.feature_importances_  
            # nonzero_fea_idxs = np.where(fea_masks)[0]
            # # Sort the feature importances in descending order  
            # sorted_indices = np.argsort(importances)[::-1]  
            # top_sorted_indices = sorted_indices[:20]
            # for i, idx in enumerate(top_sorted_indices):  
            #     fea_importance.append([feature_name[idx], importances[idx]])
            # feature_importances.append(fea_importance)

    if feature_name and len(feature_name)>0:
        return scores, feature_importances, top_idxs
    else:
        if return_prediction:
            return scores, np.exp(y_test), np.exp(y_pred)
        else:
            return scores

def tree_models_fit_and_eval(X, y, evaluators, seeds, train_frac: float= 0.7, need_normalize: bool = False, return_prediction: bool = False, return_train_loss: bool = False, train_idx=None, test_idx=None):
    scores = {}
    predictions = {}
    for seed in seeds:
        set_seed(seed)
        # Random train-test split

        if train_idx and test_idx:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            # Random train-test split
            N = int(train_frac * len(X))
            perm = np.random.permutation(len(X))
            X_train, X_test = X[perm[:N]], X[perm[N:]]
            y_train, y_test = y[perm[:N]], y[perm[N:]]


        if need_normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # n_comp_pcr = optimise_pcr_cv(X_train, y_train)
        # n_comp_plsr = optimise_pls_cv(X_train, y_train)
        rf_random = optimize_random_forest_cv(X_train, y_train)

        # alphas = np.logspace(0.001, 100, 20)
        # l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        models = {
                # 'Ridge': RidgeCV(alphas=alphas, cv=5),
                # 'Elastic net': ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=0, max_iter=100000),
                # 'PCR': make_pipeline(PCA(n_components=n_comp_pcr), LinearRegression()),
                # 'PLSR': PLSRegression(n_components=n_comp_plsr),
                'RF': RandomForestRegressor(**rf_random.best_params_)
                }
        
        model_names = list(models.keys())

        for k, model_name in enumerate(models):
            print(model_name)
            model = models[model_name]
            if seed == 0:
                scores[model_name] = []
                predictions[model_name] = []
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if return_train_loss:
                y_train_pred = model.predict(X_train)
                scores[model_name].append([[evaluator(y_train, y_train_pred), evaluator(y_test, y_pred)] for evaluator in evaluators])
            else:
                scores[model_name].append([evaluator(y_test, y_pred) for evaluator in evaluators])

            if return_prediction:
                predictions[model_name].append([np.exp(y_test), np.exp(y_pred)])

    if return_prediction:
        return scores, predictions
    else:
        return scores


def random_fit_and_eval_nn(X, y, models, evaluators, train_frac: float= 0.7, seed: int = 0, need_normalize: bool = False, need_zscore: bool = False , device: str = 'cpu', return_prediction: bool = False, return_train_loss: bool = False, train_idx=None, test_idx=None):
    set_seed(seed)

    if train_idx and test_idx:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        # Random train-test split
        N = int(train_frac * len(X))
        perm = np.random.permutation(len(X))
        X_train, X_test = X[perm[:N]], X[perm[N:]]
        y_train, y_test = y[perm[:N]], y[perm[N:]]

    # # Random train-test split
    # N = int(train_frac * len(X))
    # perm = np.random.permutation(len(X))
    # X_train, X_test = X[perm[:N]], X[perm[N:]]
    # y_train, y_test = y[perm[:N]], y[perm[N:]]


    if need_normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    # print(X_train.shape, X_test.shape)
    
    if need_zscore:
        # means = torch.mean(train_data)
        stdevs = torch.std(X_train) 
        X_train = (X_train)/stdevs
        X_test = (X_test)/stdevs

    scores = []
    predictions = []
    train_logrul = y_train
    train_mean_logrul = np.mean(train_logrul)
    train_std_logrul = np.std(train_logrul)
    train_zscore_logrul = (train_logrul - train_mean_logrul) / train_std_logrul

    for model in models:
        train_y_hat, test_y_hat = model(X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_zscore_logrul,
                    train_mean_logrul,
                    train_std_logrul)

        if return_train_loss:
            scores.append([[evaluator(y_train, train_y_hat), evaluator(y_test, test_y_hat)] for evaluator in evaluators])
        else:
            scores.append([evaluator(y_test, test_y_hat) for evaluator in evaluators])
        # scores.append([evaluator(y_test, test_y_hat) for evaluator in evaluators])
        if return_prediction:
            predictions.append([np.exp(y_test), np.exp(test_y_hat)])
    if return_prediction:
        return scores, predictions
    else:
        return scores

class MLP(nn.Module):
    def __init__(self, V, hidden_dim=256,seed=0):
        super(MLP, self).__init__()
        self.V = V
        lin1 = nn.Linear(V, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 1)
        # add this for reproducibility
        torch.manual_seed(seed)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        
    def forward(self, input):
        out = input.view(input.shape[0], self.V)
        out = self._main(out)
        return out
    
def calc_loss(pred, obs):
    loss = nn.MSELoss()
    return loss(pred,obs)
    
def MLP_rul_estimator(train_x,
                    test_x,
                    train_log_rul,
                    test_log_rul,
                    train_zscore_logrul,
                    train_mean_logrul,
                    train_std_logrul,
                    hidden_dim = 10,
                    rw = 0.0001,
                    lr = 0.001,
                    epoch=100,
                    seed=0,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, V = train_x.shape
    
    store_loss = []
    mlp = MLP(V, hidden_dim)
    mlp.to(device)
    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_minus_mean_rul = torch.from_numpy(train_log_rul - train_mean_logrul).float().to(device)
    

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    for step in range(epoch):
        preds = mlp(train_x)
        train_err = calc_loss(preds.squeeze(), train_minus_mean_rul)
        weight_norm = torch.tensor(0., device=device)

        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_err.clone()
        loss += rw * weight_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        store_loss.append(loss.item())
    
    y_train_pred = mlp(train_x).squeeze().detach().cpu().numpy()
    y_test_pred = mlp(test_x).squeeze().detach().cpu().numpy()
    
    train_y_hat = y_train_pred + train_mean_logrul
    test_y_hat = y_test_pred + train_mean_logrul

    return train_y_hat, test_y_hat
    
    


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.LazyLinear(120)
        # self.fc1 = nn.Linear(4048, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x, p=0):
#         print(x.shape)
        drop2d = nn.Dropout2d(p)
        drop1d = nn.Dropout(p)

        # Max pooling over a (2, 2) window
        x = self.pool(drop2d(F.relu(self.conv1(x))))
        # If the size is a square you can only specify a single number
        x = self.pool(drop2d(F.relu(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = drop1d(x)
        x = F.relu(self.fc2(x))
        x = drop1d(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def CNN_rul_estimator(train_x,
                    test_x,
                    train_log_rul,
                    test_log_rul,
                    train_zscore_logrul,
                    train_mean_logrul,
                    train_std_logrul,
                    n_starts = 10,
                    n_iter = 1000,
                    rw = 0.001,
                    lr = 0.0001,
                    drop_rate = 0.0,
                    seed=0):

    # best_loss = 1e10
    # best_model, best_train_y_hat, best_test_y_hat = None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x = train_x.to(device)
    test_x = test_x.to(device)
    train_minus_mean_rul = torch.from_numpy(train_log_rul - train_mean_logrul).float().to(device)

    
    train_pred_stor = []
    test_pred_stor = []


    for n in range(n_starts):
        net = CNN()
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        
        for step in range(n_iter):

            preds = net(train_x[:, None, :, :], p=drop_rate)
            train_err = calc_loss(preds.squeeze(), train_minus_mean_rul)

            weight_norm = torch.tensor(0.).to(device)
            for w in net.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_err.clone()
            loss += rw * weight_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # store_loss.append(loss.item())

        y_train_pred = net(train_x[:, None, :, :]).squeeze().detach().cpu().numpy()
        y_test_pred = net(test_x[:, None, :, :]).squeeze().detach().cpu().numpy()
        
        train_y_hat = y_train_pred + train_mean_logrul
        test_y_hat = y_test_pred + train_mean_logrul

        train_pred_stor.append(train_y_hat)
        test_pred_stor.append(test_y_hat)

    train_pred_mean = np.array(train_pred_stor).mean(axis=0)
    test_pred_mean = np.array(test_pred_stor).mean(axis=0)        
    return train_pred_mean, test_pred_mean



def optimise_pcr_cv(X, y, max_comps=20, plot_components=False):
    """
    Adapted from:
    https://www.kaggle.com/phamvanvung/principal-component-regression/code
    """
    
    # Define the PCA object
    pca = PCA()
    
    # Run PCA producing the reduced variable Xreg and select the first pc components
    Xreg_allcomps = pca.fit_transform(X)
    
    # Define components and preinitialization
    components = np.arange(1, max_comps + 1).astype('uint8')
    rmse = np.zeros((len(components), ))
    
    # Loop through all possibilities
    for comp in components:
    
        Xreg = Xreg_allcomps[:, :comp]

        # Step 2: Regression on selected components
        # Create linear regression object, fit, predict
#         lin_reg = Lasso()
        lin_reg =LinearRegression()
        Xreg_scaler = StandardScaler().fit(Xreg)
        Xreg = Xreg_scaler.transform(Xreg)

        # Cross-validation
        y_cv = cross_val_predict(lin_reg, Xreg, y, cv=5)
        y_cv = np.nan_to_num(y_cv)
        rmse[comp - 1] = mean_squared_error(10**y, 10**y_cv, squared=False)
          
    best_n_comps = np.argmin(rmse) + 1
    print(f"Suggested number of components: {best_n_comps}")
    
    if plot_components:
        with plt.style.context(('ggplot')):
            
            fig, ax = plt.subplots(figsize=(12, 4), ncols = 2)
            
            ax[0].plot(components, rmse, '-v', color = 'blue', mfc='blue')
            ax[0].plot(components[best_n_comps - 1], rmse[best_n_comps - 1], 'P', ms=10, mfc='red')
            ax[0].set_xticks(components)
            ax[0].set_xlabel('Number of PC included')
            ax[0].set_ylabel('RMSE (cycles)')
            
            ax[1].plot(components, np.cumsum(pca.explained_variance_ratio_)[:max_comps], '-v', color = 'blue', mfc='blue')
            ax[1].set_xticks(components)
            ax[1].set_xlabel('Number of PC included')
            ax[1].set_ylabel('% Variance explained')
            
            plt.tight_layout()
    
    return best_n_comps

def optimise_pls_cv(X, y, max_comps=20, plot_components=False):
    """
    Run PLS including a variable number of components, up to max_comps,
    and calculate MSE
    """
    
    components = np.arange(1, max_comps + 1).astype('uint8')
    rmse = np.zeros((len(components), ))
    
    # Loop through all possibilities
    for comp in components:
        pls = PLSRegression(n_components=comp)
        
        # Cross-validation
        y_cv = cross_val_predict(pls, X, 10**y, cv=5)


        rmse[comp - 1] = mean_squared_error(10**y, y_cv, squared=False)
    
    rmsemin = np.argmin(rmse)
    print("Suggested number of components: ", rmsemin+1)
    
    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.figure()
            plt.plot(components, rmse, '-v', color = 'blue', mfc='blue')
            plt.plot(components[rmsemin], rmse[rmsemin], 'P', ms=10, mfc='red')
            
            plt.xticks(components)
            plt.xlabel('Number of PLS components')
            plt.ylabel('RMSE (cycles)')
            plt.title('PLS')
            plt.xlim(left=-1)
    
    return rmsemin + 1

def optimize_random_forest_cv(X, y):
    
    # Define grid of hyperparameters
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    # Define random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Define model and grid
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf,
                                   param_distributions = random_grid,
                                   n_iter = 100,
                                   cv = 5,
                                   verbose = 0,
                                   random_state = 0,
                                   n_jobs = -1)

    # Fit model
    rf_random.fit(X, y)
    
    print(rf_random.best_params_)

    return rf_random



