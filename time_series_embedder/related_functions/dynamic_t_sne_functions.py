import numpy as np
import pandas as pd
import torch

from sklearn.utils import check_random_state
from typing import List, Union
from time_series_embedding.time_series_embedder.parent_embedder import transform_dataframe_to_list_array_and_check_original_data

epsilon = 1e-16
floath = np.float32


def calc_square_euclidean_norms(X: torch.Tensor, metric: str) -> torch.Tensor:
    """Calculate euclidean distance between data points if data is not distance matrix.

    Args:
        X (torch.Tenor): Data coordinate matrix.
        metric (str): Data distance metric type for calculating conditional probability.

    Returns:
        torch.Tenor: Square euclidean distance matrix.
    """
    if metric == 'precomputed':
        return X

    N = X.size()[0]
    ss = (X ** 2).sum(dim=1)

    data_distances = ss.reshape(N, 1) + ss.reshape(1, N) - 2 * torch.mm(X, torch.t(X))
    if metric == 'euclidean':
        return data_distances
    else:
        raise Exception('Invalid metric')


def calc_original_cond_prob(X: torch.Tensor, sigma: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Calculate conditional probability from distance of original data.

    Args:
        X (torch.Tensor): Data distance matrix.
        sigma (torch.Tensor): Sigma matrix for calculating conditional probability. It is computed in find sigma function.
        metric (str): Data distance metric type for calculating conditional probability.

    Returns:
        torch.Tensor: Conditional probability from distance of original data.
    """
    N = X.size()[0]
    data_distances = calc_square_euclidean_norms(X, metric)

    esqdistance = torch.exp(-data_distances / ((2 * (sigma ** 2)).reshape(N, 1)))
    esqdistance_zd = esqdistance.clone().fill_diagonal_(0)

    row_sum = torch.sum(esqdistance_zd, dim=1).reshape((N, 1))

    return esqdistance_zd / row_sum  # Possibly dangerous


def calc_original_simul_prob(original_data_tensor: torch.Tensor, sigma_tensor: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Calculate joint distribution from conditional probability of original data.

    Args:
        original_data_tensor (torch.Tensor): Data distance matrix.
        sigma_tensor (torch.Tensor): Sigma matrix for calculating conditional probability. It is computed in find sigma function.
        metric (str): Data distance metric type for calculating conditional probability.

    Returns:
        torch.Tensor: Joint distribution from conditional probability of original data.
    """
    p_Xp_given_X = calc_original_cond_prob(original_data_tensor, sigma_tensor, metric)

    return (p_Xp_given_X + torch.t(p_Xp_given_X)) / (2 * p_Xp_given_X.size()[0])


def calc_visible_simul_prob(Y: torch.Tensor) -> torch.Tensor:
    """
    Calculate joint distribution of approximate coordinates.
    Args:
        Y (torch.Tensor): Approximate coordinate matrix.

    Returns:
        torch.Tensor: joint distribution of approximate coordinates
    """
    numerators = 1 / (calc_square_euclidean_norms(Y, 'euclidean') + 1)
    numerators.fill_diagonal_(0)

    return numerators / numerators.sum()  # Possibly dangerous


def cost_var(original_data_tensor: torch.Tensor, visible_data_tensor: torch.Tensor, sigma_tensor: torch.Tensor,
             metric: str, device: Union[torch.device, str] = 'cpu') -> torch.Tensor:
    """
    Calculate KL divergence between original data distribution and approximate coordinate distribution.
    Args:
        original_data_tensor (torch.Tensor): Original data coordinate matrix.
        visible_data_tensor (torch.Tensor): Approximate data coordinate matrix.
        sigma_tensor (torch.Tensor): Sigma for calculating joint probability.
        metric (str): Data distance metric type for calculating conditional probability.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Returns:
        torch.float: KL divergence between original data distribution and approximate coordinate distribution.

    """
    epsilon_tensor = torch.tensor(epsilon).float().to(device)

    original_simul_prob = calc_original_simul_prob(original_data_tensor, sigma_tensor, metric)
    visible_simul_prob = calc_visible_simul_prob(visible_data_tensor)

    PXc = torch.maximum(original_simul_prob, epsilon_tensor)
    PYc = torch.maximum(visible_simul_prob, epsilon_tensor)

    # Possibly dangerous (clipped)
    return torch.sum(original_simul_prob * torch.log(PXc / PYc))


def find_sigma(original_data: np.ndarray, sigma: np.ndarray, N: int, perplexity: int, sigma_iters: int,
               metric: str, verbose: int = 0, device: Union[torch.device, str] = 'cpu') -> np.ndarray:
    """
    Binary search on sigma for a given perplexity.

    Args:
        original_data (np.ndarray): Original data coordinate matrix.
        sigma (np.ndarray): initial sigma matrix.
        N (int): The number of data points.
        perplexity (int): hyper parameter of variance.
        sigma_iters (int): The number of searching iteration.
        metric (str): Data distance metric type for calculating conditional probability.
        verbose (int): Whether show search processing (> 0) or not (0). Defaults to 0. If you use verbose as positive,
            the processing is shown when the number of repetitions is equal multiple of verbose.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Returns:
        np.ndarray: Result sigma.
    """
    if verbose:
        print(f'finding sigma process is using {device}')
    original_data_tensor = torch.from_numpy(original_data).clone().to(device)
    sigma_tensor = torch.from_numpy(sigma).clone().to(device)

    target = torch.tensor(np.log(perplexity)).float().to(device)
    epsilon_tensor = torch.tensor(epsilon).float().to(device)

    # Setting update for binary search interval
    sigmin = torch.from_numpy(np.full(N, np.sqrt(epsilon), dtype=floath)).to(device)
    sigmax = torch.from_numpy(np.full(N, np.inf, dtype=floath)).to(device)

    for i in range(sigma_iters):
        P = torch.maximum(calc_original_cond_prob(original_data_tensor, sigma_tensor, metric), epsilon_tensor)
        entropy = -torch.sum(P * torch.log(P), dim=1)

        sigmin = torch.where(torch.lt(entropy, target), sigma_tensor, sigmin)
        sigmax = torch.where(torch.gt(entropy, target), sigma_tensor, sigmax)

        # Setting update for sigma_tensor according to search interval
        sigma_tensor = torch.where(torch.isinf(sigmax), sigma_tensor * 2, (sigmin + sigmax) / 2.)

        if verbose and (i + 1) % 10 == 0:
            print('Iteration: {0}.'.format(i + 1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(torch.exp(torch.min(entropy)), torch.exp(torch.max(entropy))))

        if np.any(np.isnan(np.exp(entropy.to('cpu').detach().numpy().copy()))):
            raise Exception('Invalid sigmas. The perplexity is probably too low.')

    return sigma_tensor.to('cpu').detach().numpy().copy()


def movement_penalty(all_step_visible_data_tensor: torch.Tensor, N: int,
                     device: Union[torch.device, str] = 'cpu') -> torch.Tensor:
    """
    Calculate the penalty term between current step approximate coordinates and next step approximate coordinates.
    Args:
        all_step_visible_data_tensor (torch.Tensor): Approximate coordinate matrix in all steps.
        N (int): The number of data points.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu.

    Returns:
        torch.Tensor: the penalty cost.
    """
    penalties = torch.zeros(all_step_visible_data_tensor.size()[0], device=device)
    for t in range(all_step_visible_data_tensor.size()[0] - 1):
        penalties[t] = torch.sum((all_step_visible_data_tensor[t] - all_step_visible_data_tensor[t + 1]) ** 2)

    return torch.sum(penalties) / (2 * N)


def create_subtract_matrix(one_step_tensor: torch.Tensor, device: Union[torch.device, str] = 'cpu') -> torch.Tensor:
    """
    Calculate subtraction between data points in one step.
    Args:
        one_step_tensor: The coordinate of data points in one step.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Returns:
        torch.Tensor: subtracted matrix between data points in one step.
    """
    N, dims = one_step_tensor.size()
    subtract_matrix = torch.zeros(N, N, dims).float().to(device)
    for i in range(N):
        for j in range(dims):
            subtract_matrix[i, j] = one_step_tensor[i] - one_step_tensor[j]

    return subtract_matrix


def find_all_step_visible_data(all_step_original_data: np.ndarray, all_step_visible_data: np.ndarray,
                               all_step_sigmas: np.ndarray, N: int, n_steps: int, output_dims: int, n_epochs: int,
                               initial_lr: float, final_lr: float, lr_switch: int, initial_momentum: float,
                               final_momentum: float, momentum_switch: int, penalty_lambda: float, metric: str,
                               verbose: int = 0, device: Union[torch.device, str] = 'cpu') -> List[np.ndarray]:
    """
    Optimize cost wrt all_step_visible_data[t], simultaneously for all steps.

    Args:
        all_step_original_data (np.ndarray): The original coordinates of data points in all steps.
        all_step_visible_data (np.ndarray): The approximate coordinates of data points in all steps.
        all_step_sigmas (np.ndarray): The searched sigma value for calculating joint distribution in all steps.
        N (int): The number of data points.
        n_steps (int): The number of time steps.
        output_dims (int): The number of embedding dimension.
        n_epochs (int): The number of iteration for optimizing.
        initial_lr (float): The value of first phase learning rate.
        final_lr (float): The value of second phase learning rate.
        lr_switch (int): The number of iteration value when switch learning rate.
        initial_momentum (float): The value of first phase momentum.
        final_momentum (float): The value of second phase momentum.
        momentum_switch (int): The number of iteration value when switch momentum.
        penalty_lambda (float): The value of penalty weight.
        metric (str): Data distance metric type for calculating conditional probability.
        verbose (int): Whether show search processing (> 0) or not (0). Defaults to 0. If you use verbose as positive,
            the processing is shown when the number of repetitions is equal multiple of verbose.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Returns:
        List[np.ndarray]: The optimized approximate coordinates of data points in all steps.
    """

    # Optimization hyper-parameters
    lr_tensor = torch.from_numpy(np.array(initial_lr, dtype=floath)).to(device)
    momentum_tensor = torch.from_numpy(np.array(initial_momentum, dtype=floath)).to(device)

    # Penalty hyper-parameter
    penalty_lambda_tensor = torch.from_numpy(np.array(penalty_lambda, dtype=floath)).to(device)

    # Cost
    all_step_original_data_tensors = torch.from_numpy(all_step_original_data).clone().to(device)
    all_step_visible_data_tensors = torch.tensor(all_step_visible_data.copy()).float().to(device)
    all_step_visible_progress_tensors = torch.from_numpy(np.zeros((n_steps, N, output_dims), dtype=floath)).to(device)
    all_step_sigmas_tensors = torch.from_numpy(all_step_sigmas).clone().to(device)

    all_step_visible_data_tensors.requires_grad_(True)

    # Momentum-based gradient descent
    epoch = 0
    while True:
        if epoch == lr_switch:
            lr_tensor = torch.from_numpy(np.array(final_lr, dtype=floath)).to(device)
        if epoch == momentum_switch:
            momentum_tensor = torch.from_numpy(np.array(final_momentum, dtype=floath)).to(device)

        c_vars = torch.zeros(n_steps).to(device)
        for t in range(n_steps):
            c_vars[t] = cost_var(all_step_original_data_tensors[t], all_step_visible_data_tensors[t], all_step_sigmas_tensors[t],
                                 metric, device=device)

        penalty = movement_penalty(all_step_visible_data_tensors, N, device=device)
        kl_loss = torch.sum(c_vars)
        cost = kl_loss + penalty_lambda_tensor * penalty
        cost.backward()

        with torch.no_grad():

            # Setting update for all_step_visible_data velocities
            all_step_visible_progress_tensors = \
                momentum_tensor * all_step_visible_progress_tensors - lr_tensor * all_step_visible_data_tensors.grad

            # Setting update for all_step_visible_data positions
            all_step_visible_data_tensors += all_step_visible_progress_tensors
            all_step_visible_data_tensors.grad.zero_()

        if verbose and (epoch + 1) % verbose == 0:
            print(f'Epoch: {epoch + 1}. KL_loss: {kl_loss}, penalty: {penalty}')
        epoch += 1

        if epoch >= n_epochs:
            break

    final_all_step_visible_data = []

    for t in range(n_steps):
        final_all_step_visible_data.append(all_step_visible_data_tensors[t].to('cpu').detach().numpy().copy())

    return final_all_step_visible_data


def dynamic_tsne(all_step_original_df: pd.DataFrame, perplexity: int = 30,
                 all_step_visible_data: List[np.ndarray] = None, output_dims: int = 2, n_epochs: int = 1000,
                 initial_lr: float = 2400.0, final_lr: float = 200.0, lr_switch: int = 250, init_stdev: float = 1e-4,
                 sigma_iters: int = 50, initial_momentum: float = 0.5, final_momentum: float = 0.8,
                 momentum_switch: int = 250, penalty_lambda: float = 0.1, metric: str = 'euclidean',
                 random_state: Union[int, np.random.RandomState] = 0, verbose: int = 1,
                 device: Union[torch.device, str] = 'cpu') -> pd.DataFrame:
    """
    Compute sequence of projections from a sequence of matrices of observations (or distances) using dynamic t-SNE.

    Args:
        all_step_original_df (pd.DataFrame): This is consists of data coordinate or distance in each time step
            which are combined vertically. If `metric` is 'precomputed', data must be pairwise distance matrices.
        perplexity (int): Hyper parameter perplexity for binary search for sigmas. Defaults to 30.
        all_step_visible_data (List[np.ndarray]): List of matrices containing the starting positions
            for each point at each time step.
        output_dims (int): The number of embedding dimension. Defaults to 2.
        n_epochs (int): The number of iteration for optimizing. Defaults to 1000.
        initial_lr (float): The value of first phase learning rate. Defaults to 2400.0.
        final_lr (float): The value of second phase learning rate. Defaults to 200.0.
        lr_switch (int): The number of iteration value when switch learning rate. Defaults to 250.
        init_stdev (float): Standard deviation for a Gaussian distribution with zero mean from
            which the initial coordinates are sampled. Defaults to 1e-4.
        sigma_iters (int): The number of searching iteration. Defaults to 50.
        initial_momentum (float): The value of first phase momentum. Defaults to 0.5.
        final_momentum (float): The value of second phase momentum. Defaults to 0.8.
        momentum_switch (int): The number of iteration value when switch momentum. Defaults to 250'.
        penalty_lambda (float): The value of penalty weight. Defaults to 0.1.
        metric (str): Data distance metric type for calculating conditional probability. Defaults to 'euclidean'.
        random_state (Union[int, np.random.RandomState]): It is used to initialize the position of each point.
            Defaults to a random seed 0.
        verbose (int): Whether show search processing (> 0) or not (0). Defaults to 0. If you use verbose as positive,
            the processing is shown when the number of repetitions is equal multiple of verbose.
        device (torch.device or str): Using gpu or cpu information.
            If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Return:
        pd.DataFrame: DataFrame of matrices representing the sequence of projections.
            The optimized coordinates in each step are concatenated vertically.
    """
    random_state = check_random_state(random_state)
    if verbose:
        print(f'dynamic_tsne is using {device}')

    n_steps = int(all_step_original_df['step'].max() + 1)
    n_data_points = all_step_original_df.query('step == 0').shape[0]

    # transform DataFrame to list of numpy
    all_step_original_data = transform_dataframe_to_list_array_and_check_original_data(n_steps, n_data_points, all_step_original_df)

    if all_step_visible_data is None:
        initial_visible_data = random_state.normal(0, init_stdev, size=(n_data_points, output_dims))
        all_step_visible_data = [initial_visible_data] * n_steps
    # compare the number of original data points with that of visible data
    for t in range(n_steps):
        if all_step_original_data[t].shape[0] != n_data_points or all_step_visible_data[t].shape[0] != n_data_points:
            raise Exception('The number of data points does not match between given original and visible data.')

        all_step_original_data[t] = np.array(all_step_original_data[t], dtype=floath)
        all_step_visible_data[t] = np.array(all_step_visible_data[t], dtype=floath)

    if isinstance(all_step_visible_data, list):
        all_step_visible_data = np.stack(all_step_visible_data, axis=0)

    all_step_sigmas = []
    for t in range(n_steps):
        original_data = all_step_original_data[t]

        sigma = find_sigma(original_data, np.ones(n_data_points, dtype=floath), n_data_points, perplexity, sigma_iters,
                           metric, verbose=verbose, device=device)

        all_step_sigmas.append(sigma)

    all_step_sigmas = np.stack(all_step_sigmas, axis=0)
    all_step_visible_data = find_all_step_visible_data(np.stack(all_step_original_data, axis=0), all_step_visible_data,
                                                       all_step_sigmas, n_data_points, n_steps, output_dims, n_epochs,
                                                       initial_lr, final_lr, lr_switch, initial_momentum,
                                                       final_momentum, momentum_switch, penalty_lambda, metric,
                                                       verbose, device=device)

    all_step_visible_data = np.concatenate(all_step_visible_data, axis=0)
    step_array = np.array([[i] for i in range(n_steps)]).repeat(n_data_points)
    all_step_visible_data = pd.DataFrame(
        data=np.concatenate([all_step_visible_data, np.expand_dims(step_array, axis=1)], axis=1),
        columns=[f'col{i}' for i in range(output_dims)] + ['step']
    )

    return all_step_visible_data
