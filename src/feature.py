import torch
import numpy as np

from functools import partial


def annotate_labels(
    data,
    label_mode='fixed',
    label_ths: float = 1.43,
    relative_Q_cycles: int = 5,
    relative_Q_ths: float = 0.8
):
    labels = {}
    for cell_name in data:
        if label_mode == 'relative':
            label_ths = sum(
                max(x['Q_d']) for x in data[cell_name][:relative_Q_cycles]
            ) / relative_Q_cycles * relative_Q_ths
        for cycle, cycle_data in enumerate(data[cell_name]):
            if cell_name not in labels and max(cycle_data['Q_d']) < label_ths:
                labels[cell_name] = cycle + 1
    return labels


def variance_feature(
    cell_data,  # [num_samples, channels, num_cycles, dimension]
    early_cycle,
    later_cycle,
    smoother: callable = None,
    curve_differ: callable = None,
    interp_mask = None
):
    def _nanstd(x, dim, keepdim=False):
        m = x.nanmean(dim=dim, keepdim=True)
        return (x - m).square().nanmean(dim=dim, keepdim=keepdim).sqrt()

    VQ_d_early, VQ_d_later = cell_data[:, :, early_cycle], cell_data[:, :, later_cycle]
    if smoother is not None:
        VQ_d_early, VQ_d_later = smoother(VQ_d_early), smoother(VQ_d_later)

    if curve_differ is not None:
        early_mask = interp_mask[:, :, early_cycle]
        later_mask = interp_mask[:, :, later_cycle]
        diff_curve = curve_differ(VQ_d_later, VQ_d_early, later_mask, early_mask)
    else:
        diff_curve = VQ_d_later - VQ_d_early

    return _nanstd(diff_curve, dim=-1, keepdim=True).view(len(cell_data), -1)

def build_groups(i_indices, j_indices, cycle_segs):  
    groups = [f'{cycle_segs[i]} - {cycle_segs[j]}' for i, j in zip(i_indices.cpu().numpy(), j_indices.cpu().numpy()) ]  
    # for i, j in zip(i_indices.cpu().numpy(), j_indices.cpu().numpy()):  
    #     fea = f'{cycle_segs[i]} - {cycle_segs[j]}' 
    #     groups.append(fea)  
    return groups 

def pad_X(cell_data, num_signal_segments, num_cycle_groups):
    N, C, L, W = cell_data.shape

    if W % num_signal_segments !=0:
        # need padding
        new_W = (W // num_signal_segments + 1) * num_signal_segments
        padding_size_W = new_W - W
    else:
        padding_size_W = 0

    if L % num_cycle_groups !=0:
        new_L = (L // num_cycle_groups + 1) * num_cycle_groups
        padding_size_L = new_L - L
    else:
        padding_size_L = 0

    if padding_size_W > 0 or padding_size_L > 0:
        cell_data = torch.nn.functional.pad(cell_data, (0, padding_size_W,  0, padding_size_L))

    return cell_data

def our_feature(
    cell_data,  # [num_samples, channels, num_cycles, dimension]
    num_cycle_groups: int,
    num_signal_segments: int,
    aggregators: list[callable],
    activators: list[callable] = None,
    smoothers: list[callable] = None,
    device: str = 'cpu',
    curve_differ: callable = None,
    build_feature_name: bool = False
):
    if not isinstance(aggregators, list):
        aggregators = [aggregators]
    if smoothers is not None and not isinstance(smoothers, list):
        smoothers = [smoothers]
    if activators is not None and not isinstance(activators, list):
        activators = [activators]

    if smoothers is not None:
        cell_data = torch.concat([smoother(cell_data) for smoother in smoothers], dim=1)

    cell_data = pad_X(cell_data, num_signal_segments, num_cycle_groups)

    # Reshape such that the cycle groups and feature segments are ready
    N, C, L, _ = cell_data.shape
    cell_data = cell_data.reshape(N, C, L, num_signal_segments, -1)
    E = cell_data.shape[-1]
    cell_data = cell_data.reshape(N, C, num_cycle_groups, -1, num_signal_segments, E)
    
    
    # Use torch to accelerate
    cell_data = cell_data.to(device)
    # Aggregate the cycle_group dimension
    cell_data = torch.cat([partial(agg, dim=3)(cell_data) for agg in aggregators], dim=1)
    # Calculate the mutual difference between cycle groups
    indices = torch.triu_indices(num_cycle_groups, num_cycle_groups, offset=1, device=device)
    i_indices, j_indices = indices[0], indices[1]
    if curve_differ is not None:
        diff_data = curve_differ(cell_data[:, :, i_indices], cell_data[:, :, j_indices])
    else:
        diff_data = cell_data[:, :, i_indices] - cell_data[:, :, j_indices]
    cell_data = torch.concat([cell_data, diff_data], dim=2)
    # Aggregate the signal dimension
    cell_data = torch.cat([partial(agg, dim=-1)(cell_data) for agg in aggregators], dim=1)
    # Activation if provided
    if activators is not None:
        cell_data = torch.cat([act(cell_data) for act in activators], dim=1)

    cell_data = cell_data.view(N, -1).cpu().numpy()
    
    if build_feature_name:
        channels = ['VQ_c', 'VQ_d', 'dVdQ_c', 'dVdQ_d', 'I_c', 'I_d','V_c', 'V_d', 'QV_c', 'QV_d', 'E_c', 'E_d', 'W_c',  'W_d']
        # channels = ['VQ_d']
        cycle_segs = [f'Cycle({i+1}/{num_cycle_groups})' for i in range(num_cycle_groups)]
        signal_segs = [f'{i+1}/{num_signal_segments}' for i in range(num_signal_segments)]

        # generate group feature name
        diff_feature_name = build_groups(i_indices, j_indices, cycle_segs)
        cycle_segs.extend(diff_feature_name) # 15

        activator_names = []
        if activators is not None:
            for act in activators:
                name = 'identity' if act.__name__ == '<lambda>' else act.__name__
                activator_names.append(name)

        fea_names = [f'{act}({agg_cycle.__name__}({cycle_seg})[{agg_seg.__name__}({channel}({signal_seg}))])'
            for act in activator_names
            for agg_seg in aggregators
            for agg_cycle in aggregators
            for channel in channels
            for cycle_seg in cycle_segs
            for signal_seg in signal_segs
        ]


        return cell_data, fea_names
    else:
        return cell_data

