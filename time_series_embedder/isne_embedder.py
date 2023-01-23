import numpy as np
import pandas as pd
import sys
import torch

from time_series_embedding.time_series_embedder.parent_embedder import ParentTimeSeriesEmbedder, \
    transform_dataframe_to_list_array_and_check_original_data
from time_series_embedding.config.time_series_config import param_to_data_key
from typing import List, Union

EPS = sys.float_info.epsilon


# ToDO: docstringの追加
class IsneEmbedder(ParentTimeSeriesEmbedder):

    def __init__(self, *args) -> None:
        super(IsneEmbedder, self).__init__(*args)
        self.class_key = param_to_data_key('I_SNE', self.params)

    def exec_embed(self, embed_dim: int = 3, device: Union[torch.device, str] = 'cpu') -> None:
        """
        Compute embedded coordinates using Inner_SNE.
        Args:
            embed_dim (int): The number of embedding dimension.
            device (torch.device or str): Using gpu or cpu information.
                If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

        Returns:
            None
        """
        EPS_tensor = torch.tensor(EPS).to(device)
        all_step_original_data_list = transform_dataframe_to_list_array_and_check_original_data(self.n_steps, self.n_data_points, self.df)
        if all_step_original_data_list[0].shape[0] != all_step_original_data_list[0].shape[1]:
            # prepare distance mat at each steps
            all_step_original_data_list = calc_euclid_distance_matrix(all_step_original_data_list, device)
        # prepare stochastic series
        all_step_original_probabilities = calc_original_probabilities(all_step_original_data_list, device)

        # calculate y_fix
        y_fix = self.calc_y_fix(embed_dim, all_step_original_probabilities, self.params["n_iterations"],
                                self.params["lr"], self.params['verbose'], device, EPS_tensor
                                )
        fitted_coordinate_list = self.calc_all_step_coordinates(embed_dim, y_fix, all_step_original_probabilities,
                                                                self.params["n_iterations"], self.params["lr"],
                                                                self.params["internal_weight"], self.params['verbose'],
                                                                device, EPS_tensor)
        fitted_coord_array = np.concatenate(fitted_coordinate_list, axis=0)
        step_array = np.array([[i] for i in range(self.n_steps)]).repeat(self.n_data_points)

        self.coordinates = pd.DataFrame(
            data=np.concatenate([fitted_coord_array, np.expand_dims(step_array, axis=1)], axis=1),
            columns=[f'col{i}' for i in range(embed_dim)] + ['step']
        )

    def calc_y_fix(self, embed_dim: int, all_step_original_probabilities: List[np.ndarray], n_iterations: int,
                   lr: float, verbose: int, device: Union[torch.device, str], EPS_tensor: torch.tensor) -> np.ndarray:
        """

        Args:
            embed_dim:
            all_step_original_probabilities:
            n_iterations:
            lr:
            verbose (int): Whether show search processing (> 0) or not (0). Defaults to 0. If you use verbose as positive,
                the processing is shown when the number of repetitions is equal multiple of verbose.
            device:
            EPS_tensor:

        Returns:

        """
        average_probability = torch.from_numpy(sum(all_step_original_probabilities) / self.n_steps).to(device)
        # prepare initial fitted data
        x = torch.normal(mean=0.0, std=1.0, size=(self.n_data_points, embed_dim)).to(device)
        y = torch.normal(mean=0.0, std=1.0, size=(self.n_data_points, embed_dim)).to(device)
        x.requires_grad_(True)
        y.requires_grad_(True)
        # optimize kl divergence about x and y
        if verbose:
            print(f'optimizing y_fix using device: {device}')
        for i in range(n_iterations):
            approximate_probability = calc_probability(x, y)
            kl_loss = calc_kl_divergence(average_probability, approximate_probability, EPS_tensor)
            torch.autograd.set_detect_anomaly(True)
            kl_loss.backward()
            with torch.no_grad():
                x -= x.grad * lr
                y -= y.grad * lr
                x.grad.zero_()
                y.grad.zero_()
            if verbose and (i + 1) % verbose == 0:
                print(f'{i + 1}epoch| y_fix kl_loss: {kl_loss.to("cpu").detach().numpy().copy()}')

        return y.to('cpu').detach().numpy().copy()

    def calc_all_step_coordinates(self, embed_dim: int, y_fix: np.ndarray, all_step_original_probabilities: List[np.ndarray],
                                  n_iterations: int, lr: float, internal_weight: float, verbose: int,
                                  device: Union[torch.device, str], EPS_tensor: torch.tensor) -> List[np.ndarray]:
        """

        Args:
            embed_dim:
            y_fix:
            all_step_original_probabilities:
            n_iterations:
            lr:
            internal_weight:
            verbose (int): Whether show search processing (> 0) or not (0). Defaults to 0. If you use verbose as positive,
                the processing is shown when the number of repetitions is equal multiple of verbose.
            device:
            EPS_tensor:

        Returns:

        """
        all_step_coord_list = []
        all_step_approximate_probability_list = []
        if verbose:
            print(f'optimizing coordinates using device: {device}')
        for step in range(self.n_steps):
            print(f'<{step} step coordinates optimization>')
            x = torch.normal(mean=0.0, std=1.0, size=(self.n_data_points, embed_dim)).to(device)
            step_probability_tensor = torch.from_numpy(all_step_original_probabilities[step]).to(device)
            y_fix_tensor = torch.from_numpy(y_fix).to(device)
            x.requires_grad_(True)
            if step == 0:
                for i in range(n_iterations):
                    approximate_probability = calc_probability(x, y_fix_tensor)
                    kl_loss = calc_kl_divergence(step_probability_tensor, approximate_probability, EPS_tensor)
                    kl_loss.backward()
                    with torch.no_grad():
                        x -= x.grad * lr
                        x.grad.zero_()
                    if verbose and (i + 1) % verbose == 0:
                        print(f'{i + 1}epoch| p_q_kl_loss: {kl_loss.to("cpu").detach().numpy().copy()}')
            else:
                for i in range(n_iterations):
                    approximate_probability = calc_probability(x, y_fix_tensor)
                    p_q_kl_loss = internal_weight * calc_kl_divergence(step_probability_tensor, approximate_probability,
                                                                       EPS_tensor)
                    q_q_kl_loss = (1 - internal_weight) * calc_kl_divergence(
                        all_step_approximate_probability_list[step - 1], approximate_probability, EPS_tensor
                    )
                    kl_loss = p_q_kl_loss + q_q_kl_loss
                    kl_loss.backward()
                    with torch.no_grad():
                        x -= x.grad * lr
                        x.grad.zero_()
                    if verbose and (i + 1) % verbose == 0:
                        print(f'{i + 1}epoch| p_q_kl_loss: {p_q_kl_loss.to("cpu").detach().numpy().copy()},'
                              f' q_q_kl_loss: {q_q_kl_loss.to("cpu").detach().numpy().copy()}')

            all_step_approximate_probability_list.append(calc_probability(x, y_fix_tensor).clone().detach())
            all_step_coord_list.append(x.to('cpu').detach().numpy().copy())

        return all_step_coord_list


def calc_euclid_distance_matrix(all_step_coords: List[np.ndarray],
                                device: Union[torch.device, str] = 'cpu') -> List[np.ndarray]:
    """
    Calculate euclidean distances between data coordinates.
    Args:
        all_step_coords (List[np.ndarray]): coordinate array in all steps.
        device (torch.device or str): Using gpu or cpu information.
                If use str, device == 'cpu' is only allowed when using cpu. Defaults to 'cpu'.

    Returns:
        all_step_distance_matrices (List[np.ndarray]): The euclidean distance matrix between data coordinates.
    """
    all_step_distance_matrices = []
    for coords in all_step_coords:
        coords_tensor = torch.from_numpy(coords).float().to(device)
        distance_matrix_tensor = torch.cdist(coords_tensor, coords_tensor, p=2)
        all_step_distance_matrices.append(distance_matrix_tensor.to('cpu').detach().numpy().copy())

    return all_step_distance_matrices


def calc_original_probabilities(all_step_data_distances: List[np.ndarray],
                                device: Union[torch.device, str] = 'cpu') -> List[np.ndarray]:
    """

    Args:
        all_step_data_distances:
        device:

    Returns:

    """
    all_step_probability_list = []
    for distance in all_step_data_distances:
        distance_tensor = torch.from_numpy(distance).float().to(device)
        distance_tensor = torch.exp(-1 * distance_tensor)
        distance_tensor.fill_diagonal_(0)
        probability = distance_tensor / torch.sum(distance_tensor, dim=0)
        all_step_probability_list.append(probability.to('cpu').detach().numpy().copy())

    return all_step_probability_list


def calc_probability(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """

    Args:
        x:
        y:

    Returns:

    """
    inner_x_y = torch.exp(torch.mm(x, torch.t(y)))
    # inner_x_y.fill_diagonal_(0)
    inner_x_y_diagonal = torch.diagonal(inner_x_y.clone().detach())
    inner_x_y_diagonal_matrix = torch.diag(inner_x_y_diagonal)
    inner_x_y = inner_x_y - inner_x_y_diagonal_matrix

    probability_q = inner_x_y / torch.sum(inner_x_y, dim=0)

    return probability_q


def calc_kl_divergence(p: torch.Tensor, q: torch.Tensor, EPS_tensor: torch.tensor) -> torch.float32:
    """

    Args:
        p:
        q:
        EPS_tensor:

    Returns:

    """
    p_eps = torch.maximum(p, EPS_tensor)
    q_eps = torch.maximum(q, EPS_tensor)
    kl_value = torch.mul(p, torch.log(p_eps / q_eps))

    return torch.sum(kl_value)
