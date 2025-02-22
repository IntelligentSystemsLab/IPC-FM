# -*- coding: UTF-8 -*-
"""
@filename:data_manager.py
@author:Chen Kunxu
@Time:2023/8/5
"""
import numpy as np
import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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
    Q_all = np.array([DoW, age, female, driving_license, bus_scale, car_ownership, faretype, purpose, fueltype, start_time,travel_month, travel_year, bus_interchange, distance])
    
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

        # if model_name == "MNL":
        #     Q_house = np.tile(Q_house[:, :, np.newaxis], reps=(1, 1, 4))
        #     X_house = np.concatenate((Q_house, X_house), axis=1)
        #     X.append(X_house)
        #     Q.append(Q_house)
        # else:
        X.append(X_house)
        Q.append(Q_house)
    return X, Q, saqure


def load_data(unique_catagory,model_name):
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = pd.read_csv(filePath + 'LPMC_process' + '.csv')
    main_data, extra_data, saqure = process_data(data, unique_catagory,model_name)
    return main_data, extra_data, saqure

def load_data_totest(unique_catagory,model_name, saqure):
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = pd.read_csv(filePath + 'LPMC_process' + '.csv')
    main_data, extra_data, _ = process_data(data, unique_catagory,model_name, saqure)
    return main_data, extra_data


def swiss_load_client(filePath, fileInputName, filePart='',
                      simpleArchitecture=False, lmnlArchitecture=False, write=True, split_type=0, split_num=1):
    """
    目标：生成每种目标的总的数据集 return一个数组 users_all[id]为每个人的所有数据
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives
    The first input is the X feature set, it ressembles the utility functions.
        - The shape is (n x betas+1 x alternatives), where the added +1 is the label.
    The second input is the Q feature set.
        - The shape is (n x Q_features x 1)
    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function, only TT, COST, HE
    simpleArchitecture = False & lmnlArchitecture = False :ASC ，TT, COST, HE, GA, AGE, LUGGAGE, SEATS, CHOICE
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :param write:           Save X and Q inputs in a .npy
    :return:    main_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
    """
    extend = ''
    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'

    train_data_name = filePath + fileInputName + extend + filePart + '.npy'  # 生成训练集

    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = np.loadtxt(filePath + 'swissmetro' + filePart + '.dat', skiprows=1)  # new是重新切分的

    # 集中处理数据

    CHOICE = data[:, -1]
    PURPOSE = data[:, 4]
    CAR_AV = data[:, 16]
    TRAIN_AV = data[:, 15]
    SM_AV = data[:, 17]

    # 删除AV==0 或 CHOICE ==0 的people
    exclude = ((CAR_AV == 0) + (CHOICE == 0) + (TRAIN_AV == 0) + (SM_AV == 0)) > 0
    exclude_list = [i for i, k in enumerate(exclude) if k > 0]

    data = np.delete(data, exclude_list, axis=0)

    ID = np.unique(data[:, 3])
    people_num = len(ID)
    data_num = len(data)

    total_data = data  # [9036*28]
    total_data = np.array(np.split(total_data, people_num))
    if split_type == 0:
        CHOICE = total_data[:, :, -1]  # 1004*9
        TRAIN_TT = total_data[:, :, 18]
        TRAIN_COST = total_data[:, :, 19] * (total_data[:, :, 12] == 0)  # if he owns a GA 如果有GA cost为0
        SM_TT = total_data[:, :, 21]
        SM_COST = total_data[:, :, 22] * (total_data[:, :, 12] == 0)  # if he owns a GA
        CAR_TT = total_data[:, :, 25]
        CAR_CO = total_data[:, :, 26]

        TRAIN_HE = total_data[:, :, 20]
        SM_HE = total_data[:, :, 23]
        GA = total_data[:, :, 12]
        AGE = total_data[:, :, 9]

        LUGGAGE = total_data[:, :, 8]
        SM_SEATS = total_data[:, :, 24]

        scale = 100.0

        TRAIN_TT_SCALED = TRAIN_TT / scale
        TRAIN_COST_SCALED = TRAIN_COST / scale
        SM_TT_SCALED = SM_TT / scale
        SM_COST_SCALED = SM_COST / scale
        CAR_TT_SCALED = CAR_TT / scale
        CAR_CO_SCALED = CAR_CO / scale
        TRAIN_HE_SCALED = TRAIN_HE / scale
        SM_HE_SCALED = SM_HE / scale

        ASCs = np.ones((people_num, int(data_num / people_num)))
        ZEROs = np.zeros((people_num, int(data_num / people_num)))

        CHOICE_CAR = (CHOICE == 3)
        CHOICE_SM = (CHOICE == 2)
        CHOICE_TRAIN = (CHOICE == 1)

        main_data = np.array(
            [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, GA, AGE, ZEROs, ZEROs, CHOICE_TRAIN],
             [ZEROs, ASCs, SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, GA, ZEROs, ZEROs, SM_SEATS, CHOICE_SM],
             [ASCs, ZEROs, CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, ZEROs, ZEROs, LUGGAGE, ZEROs, CHOICE_CAR]])
        # 前两项用于构建ASC ，TT, COST, HE, GA, AGE, LUGGAGE, SEATS, CHOICE 输入线性模型 X1 7+2(ASC)个特征输入线性

        if simpleArchitecture:
            main_data = np.array(
                [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
                 [ZEROs, ASCs, SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, CHOICE_SM],
                 [ASCs, ZEROs, CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, CHOICE_CAR]])
        # 前两项用于构建ASC ，TT, COST, HE, CHOICE 输入线性模型 X2 7个特征输入非线性

        if lmnlArchitecture:
            main_data = np.array(
                [[TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
                 [SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, CHOICE_SM],
                 [CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, CHOICE_CAR]])
        # TT, COST, HE, CHOICE 输入线性模型 没有ASC X2 3个特征输入线性

        main_data = np.swapaxes(np.swapaxes(np.swapaxes(main_data, 0, 2), 1, 3), 2, 3)  # (people, menu, beta, choice)

        if simpleArchitecture or lmnlArchitecture:
            # Hybrid Simple
            extra_data = np.delete(total_data, [18, 19, 21, 22, 25, 26, 27, 20, 23, 0, 1, 2, 3, 15, 16, 17], axis=2)
            # if simpleArchitecture:
            extra_data[:, :, 7][extra_data[:, :, 7] == 0] = 1
            # 18,19,21,22,25,26分别是TT,CO; 20,23是HE X2线性部分的特征；27是CHOICE；0,1,2,3是GROUP,SURVEY,SP,ID；15, 16,17是AV
            # 保留12个特征
        else:
            # Hybrid MNL
            extra_data = np.delete(total_data,
                                   [18, 19, 21, 22, 25, 26, 27, 20, 23, 0, 1, 2, 3, 8, 9, 12, 24, 15, 16, 17],
                                   axis=2)
            # 8, 9, 12, 24是 LUGGAGE,AGE,GA,SEATS
            # 保留8个特征 X1
            # (people, menu, feature)
        print(main_data.shape, extra_data.shape)
    return main_data, extra_data


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
