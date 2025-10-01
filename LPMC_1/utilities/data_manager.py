# -*- coding: UTF-8 -*-
"""
@filename:data_manager.py
@author:Chen Kunxu
@Time:2023/8/5
"""
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def add_positional_gaussian_noise(data, positions, noise_std=0.0, seed=None):
    """
    为3D numpy矩阵的指定位置添加高斯噪声
    
    Args:
        data: 3D numpy数组，shape为(n_samples, height, width)
        positions: 位置列表，每个位置为(row, col)元组
        noise_std: 高斯噪声的标准差，为0时不添加噪声
        seed: 随机种子，用于确保结果可重现
    
    Returns:
        添加噪声后的数据（原数据的副本）
        
    Example:
        # 对shape为(10, 6, 4)的数据在指定位置添加噪声
        positions = [(4, 0), (4, 1), (4, 2), (5, 2), (4, 3), (5, 3)]
        noisy_data = add_positional_gaussian_noise(data, positions, noise_std=0.1)
    """
    if noise_std <= 0 or len(positions) == 0:
        return data.copy()
    
    if seed is not None:
        np.random.seed(seed)
    
    # 创建数据副本，避免修改原数据
    noisy_data = data.copy()
    
    # 为每个样本在指定位置添加噪声
    for sample_idx in range(data.shape[0]):
        for row, col in positions:
            # 检查位置是否在数据范围内
            if 0 <= row < data.shape[1] and 0 <= col < data.shape[2]:
                noise = np.random.normal(0, noise_std)
                noisy_data[sample_idx, row, col] += noise
            else:
                print(f"警告: 位置 ({row}, {col}) 超出数据范围 {data.shape}")
    
    return noisy_data


def add_gaussian_noise(data, noise_std=0.0, seed=None):
    """
    为数据添加高斯噪声
    
    Args:
        data: 需要添加噪声的数据（numpy array）
        noise_std: 高斯噪声的标准差，为0时不添加噪声
        seed: 随机种子，用于确保结果可重现
    
    Returns:
        添加噪声后的数据
    """
    if noise_std <= 0:
        return data
    
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, noise_std, data.shape)
    return data + noise


def add_label_noise(data, noise_rate=0.0, seed=None):
    """
    为分类标签添加噪声（模拟错误标注）
    
    Args:
        data: 分类数据（numpy array）
        noise_rate: 噪声比例，0-1之间，表示有多少比例的标签会被随机改变
        seed: 随机种子
    
    Returns:
        添加噪声后的数据
    """
    if noise_rate <= 0:
        return data
    
    if seed is not None:
        np.random.seed(seed)
    
    data_noisy = data.copy()
    n_samples = len(data)
    n_noise = int(n_samples * noise_rate)
    
    # 随机选择要添加噪声的样本
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    unique_labels = np.unique(data)
    
    # 为选中的样本随机分配不同的标签
    for idx in noise_indices:
        # 排除当前标签，从其他标签中随机选择
        other_labels = unique_labels[unique_labels != data[idx]]
        if len(other_labels) > 0:
            data_noisy[idx] = np.random.choice(other_labels)
    
    return data_noisy


def add_missing_data(data, missing_rate=0.0, missing_type="random", seed=None):
    """
    为数据添加缺失值（模拟数据不完整场景）
    
    Args:
        data: 输入数据 (numpy array)
        missing_rate: 缺失比例 0-1之间
        missing_type: 缺失类型 ("random", "feature", "sample", "pattern")
        seed: 随机种子
    
    Returns:
        包含缺失值的数据，缺失值用np.nan表示
    """
    if missing_rate <= 0:
        return data
    
    if seed is not None:
        np.random.seed(seed)
    
    data_missing = data.copy().astype(float)  # 转换为float以支持np.nan
    
    # 随机缺失：在整个数据中随机选择位置设为缺失
    mask = np.random.rand(*data.shape) < missing_rate
    data_missing[mask] = np.nan

    return data_missing

def handle_missing_data(data, method="mean", seed=None):
    """
    处理缺失数据
    
    Args:
        data: 包含缺失值的数据
        method: 处理方法 ( "mean", "mode")
        seed: 随机种子
    
    Returns:
        处理后的数据
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_processed = data.copy()
    
    if method == "mean":
        # 用均值填充（仅适用于数值特征）
        for col_idx in range(data_processed.shape[1]):
            col_data = data_processed[:, col_idx]
            if np.isnan(col_data).any():
                mean_val = np.nanmean(col_data)
                data_processed[np.isnan(col_data), col_idx] = mean_val
                
    elif method == "mode":
        # 用众数填充（适用于分类特征）
        for col_idx in range(data_processed.shape[1]):
            col_data = data_processed[:, col_idx]
            if np.isnan(col_data).any():
                # 计算众数
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    unique_vals, counts = np.unique(valid_data, return_counts=True)
                    mode_val = unique_vals[np.argmax(counts)]
                    data_processed[np.isnan(col_data), col_idx] = mode_val

    return data_processed


def add_demographic_noise(data_ori, label_noise_rate=0.0, seed=None):
    """
    为个人信息特征添加噪声（模拟调研中的误差）
    
    Args:
        data_ori: 原始数据DataFrame
        noise_std: 连续特征的高斯噪声标准差
        label_noise_rate: 分类特征的标签噪声比例
        seed: 随机种子
    
    Returns:
        添加噪声后的数据DataFrame
    """
    if label_noise_rate <= 0:
        return data_ori
    
    data = data_ori.copy()
    
    if seed is not None:
        np.random.seed(seed)

   # 为分类型个人特征添加标签噪声
    if label_noise_rate > 0:
        # 性别（可能存在隐私考虑导致的错误申报）
        data[:, 2] = add_label_noise(data[:, 2], label_noise_rate * 0.2, seed)  # 性别噪声较小
        
        # 驾驶执照（可能存在虚假申报）
        data[:, 3] = add_label_noise(data[:, 3], label_noise_rate * 0.2, seed)

        # 车辆拥有情况（可能存在社会期望偏差）
        data[:, 5] = add_label_noise(data[:, 5], label_noise_rate * 0.2, seed)

         # 票价类型（可能存在记忆偏差）
        data[:, 6] = add_label_noise(data[:, 6], label_noise_rate * 0.2, seed)

        # 出行目的（可能存在记忆偏差或隐私考虑）
        data[:, 7] = add_label_noise(data[:, 7], label_noise_rate * 0.2, seed)

        # 燃料类型（可能不确定或记忆模糊）
        data[:, 8] = add_label_noise(data[:, 8], label_noise_rate * 0.2, seed)
    
    return data


def process_data(data_ori, unique_catagory, model_name, saqure=None):
    X = []
    Q = []
    
    household_id = data_ori['household_id'].values
    DoW = data_ori['day_of_week'].values
    age = (data_ori['age'].values / 20).astype(int)
    female = data_ori['female'].values
    driving_license = data_ori['driving_license'].values
    bus_scale = data_ori['bus_scale'].values
    car_ownership = data_ori['car_ownership'].values
    faretype = data_ori['faretype'].values
    purpose = data_ori['purpose'].values
    fueltype = data_ori['fueltype'].values
    start_time = (data_ori['start_time'].values / 6).astype(int)
    travel_month = data_ori['travel_month'].values
    travel_year = data_ori['travel_year'].values-2012
    bus_interchange = data_ori['pt_n_interchanges'].values
    distance = (data_ori['distance'].values / 1000).astype(int)
    unique = [set(DoW), set(age), set(female), set(driving_license), set(bus_scale), set(car_ownership),
              set(faretype), set(purpose), set(fueltype), set(start_time), set(travel_month), set(travel_year),
              set(bus_interchange), set(distance)]
    Q_all = np.array([DoW, age, female, driving_license, bus_scale, car_ownership, faretype, purpose, fueltype, 
                      start_time,travel_month, travel_year, bus_interchange, distance])
    
    if model_name == 'L_MNL' or model_name == "ASU_DNN" or model_name == 'MNL' :
        Q_all = (Q_all - Q_all.mean(axis=0)) / (Q_all.std(axis=0))

    else:
        Label_list = np.array(range(0, unique_catagory, 1))
        np.random.shuffle(Label_list)
        x_length = 0
        for m in unique:
            Max_m = max(list(m))
            if x_length < Max_m:
                x_length = Max_m
        if saqure is None:
            saqure = np.zeros((len(unique), x_length + 1))
        m = 0
        for i in range(0, len(unique)):
            for j in list(unique[i]):
                saqure[i, int(j)] = Label_list[m]
                m += 1
        x_0, y_0 = Q_all.shape
        for j in range(0, y_0):
            for i in range(0, x_0):
                Q_all[i, j] = saqure[i, int(Q_all[i, j])]

    for house_id in set(household_id):
        data = data_ori[data_ori['household_id'] == house_id]
        CHOICE = data['travel_mode'].values
        CHOICE_car = (CHOICE == 3)
        CHOICE_PT = (CHOICE == 2)
        CHOICE_cycling = (CHOICE == 1)
        CHOICE_WALK = (CHOICE == 0)

        TT_walking = data['dur_walking'].values
        TT_cycling = data['dur_cycling'].values
        TT_car = data['dur_driving'].values
        TT_PT = data['dur_pt_total'].values

        Cost_car = data['cost_driving_total'].values
        Cost_PT = data['cost_transit'].values

        ASCs = np.ones(CHOICE.size)
        ZEROs = np.zeros(CHOICE.size)
        X_house = np.array(
            [[ZEROs, ZEROs, ZEROs, TT_walking, ZEROs, CHOICE_WALK],
             [ASCs, ZEROs, ZEROs, TT_cycling, ZEROs, CHOICE_cycling],
             [ZEROs, ASCs, ZEROs, TT_PT, Cost_PT, CHOICE_PT],
             [ZEROs, ZEROs, ASCs, TT_car, Cost_car, CHOICE_car]])
        X_house = np.swapaxes(X_house, 0, 2)

        target_col_index = np.where(household_id == house_id)[0]
        Q_house = Q_all[:, target_col_index]
        Q_house = np.swapaxes(Q_house, 0, 1)

        X.append(X_house)
        Q.append(Q_house)
    return X, Q, saqure


def load_data(unique_catagory, model_name):
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = pd.read_csv(filePath + 'LPMC_process' + '.csv')
    main_data, extra_data, saqure = process_data(data, unique_catagory, model_name)
    return main_data, extra_data, saqure

class myData(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
