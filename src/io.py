import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path

from src.utils import calc_capacity, calc_V_of_Q, calc_dV_over_dQ, interpolate_element

def load_data(root_dir, is_ours=True, interp_dims=1000, Q_min=0, Q_max=2.2, V_min=2.5, V_max=4.2):
    def calc_delta(x):
        x = x - np.roll(x, 1)
        x[0] = 0.
        return x
    data = {}
    # max_V, min_V = 0, 10
    
    for cell_file in tqdm(list(Path(root_dir).glob('*.pkl')), desc='Loading cells'):
        cell_name = cell_file.stem
        with open(cell_file, 'rb') as f:
            cell_data = pickle.load(f)
        for cycle, cycle_data in enumerate(cell_data):
        #     if np.max(cycle_data['Q_d']) > max_V:
        #         max_V = np.max(cycle_data['Q_d'])
        #     if np.max(cycle_data['Q_c']) > max_V:
        #         max_V = np.max(cycle_data['Q_c'])
        #     if np.min(cycle_data['Q_d']) < min_V:
        #         min_V = np.min(cycle_data['Q_d'])
        #     if np.min(cycle_data['Q_c']) < min_V:
        #         min_V = np.min(cycle_data['Q_c'])

            if is_ours:
                cycle_data['QV_d'] = interpolate_element(cycle_data['Q_d(V_d)'][0], cycle_data['Q_d(V_d)'][1], device='cpu', inter_num=interp_dims)
                cycle_data['QV_c'] = interpolate_element(cycle_data['Q_c(V_c)'][0], cycle_data['Q_c(V_c)'][1], device='cpu', inter_num=interp_dims)

                cycle_data['interp_E_c'] = interpolate_element(cycle_data['t_c'], cycle_data['E_c'], device='cpu', inter_num=interp_dims)
                cycle_data['interp_W_c'] = interpolate_element(cycle_data['t_c'], cycle_data['W_c'], device='cpu', inter_num=interp_dims)
                cycle_data['interp_E_d'] = interpolate_element(cycle_data['t_d'], cycle_data['E_d'], device='cpu', inter_num=interp_dims)
                cycle_data['interp_W_d'] = interpolate_element(cycle_data['t_d'], cycle_data['W_d'], device='cpu', inter_num=interp_dims)

            else:
                if len(cycle_data['V_d']) !=0 and len(cycle_data['V_c']) !=0:
                    cycle_data['QV_d'], _ =  calc_V_of_Q(cycle_data['Q_d'], cycle_data['V_d'], Q_min=V_min, Q_max=V_max, interp_dims=interp_dims)
                    cycle_data['QV_c'], _ =  calc_V_of_Q(cycle_data['Q_c'], cycle_data['V_c'], Q_min=V_min, Q_max=V_max, interp_dims=interp_dims)
                if 'E_c' in cycle_data and 'E_d' in cycle_data:
                    cycle_data['interp_E_c'] = interpolate_element(cycle_data['t_c'], cycle_data['E_c'], device='cpu', inter_num=interp_dims)
                    cycle_data['interp_E_d'] = interpolate_element(cycle_data['t_d'], cycle_data['E_d'], device='cpu', inter_num=interp_dims)

                if 'W_c' in cycle_data and 'W_d' in cycle_data:
                    cycle_data['interp_W_c'] = interpolate_element(cycle_data['t_c'], cycle_data['W_c'], device='cpu', inter_num=interp_dims)
                    cycle_data['interp_W_d'] = interpolate_element(cycle_data['t_d'], cycle_data['W_d'], device='cpu', inter_num=interp_dims)

            cycle_data['Q_c'] = calc_capacity(cycle_data['I_c'], cycle_data['t_c'])
            cycle_data['Q_d'] = calc_capacity(cycle_data['I_d'], cycle_data['t_d'])

            if len(cycle_data['V_d']) !=0 and len(cycle_data['V_c']) !=0:
                cycle_data['VQ_c'], cycle_data['VQ_c_mask'] = calc_V_of_Q(
                    cycle_data['V_c'], cycle_data['Q_c'], interp_dims=interp_dims, Q_min=Q_min, Q_max=Q_max,
                )
                cycle_data['VQ_d'], cycle_data['VQ_d_mask'] = calc_V_of_Q(
                    cycle_data['V_d'], cycle_data['Q_d'], interp_dims=interp_dims, Q_min=Q_min, Q_max=Q_max,
                )
                cycle_data['dVdQ_c'], _ = calc_V_of_Q(
                    calc_dV_over_dQ(cycle_data['V_c'], cycle_data['Q_c']), cycle_data['Q_c'], interp_dims=interp_dims, Q_min=Q_min, Q_max=Q_max,
                )
                cycle_data['dVdQ_d'], _ = calc_V_of_Q(
                    calc_dV_over_dQ(cycle_data['V_d'], cycle_data['Q_d']), cycle_data['Q_d'], interp_dims=interp_dims, Q_min=Q_min, Q_max=Q_max,
                )

                cycle_data['interp_V_c'] = interpolate_element(cycle_data['t_c'], cycle_data['V_c'], inter_num=interp_dims)
                cycle_data['interp_I_c'] = interpolate_element(cycle_data['t_c'], cycle_data['I_c'], inter_num=interp_dims)
                cycle_data['interp_V_d'] = interpolate_element(cycle_data['t_d'], cycle_data['V_d'], inter_num=interp_dims)
                cycle_data['interp_I_d'] = interpolate_element(cycle_data['t_d'], cycle_data['I_d'], inter_num=interp_dims)
            else:
                print(cell_name, cycle)
                
            

        data[cell_name] = cell_data[1:]
        # break
        

    # print(max_V, min_V)
    return data
