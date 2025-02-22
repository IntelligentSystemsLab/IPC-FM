# -*- coding: UTF-8 -*-
"""
@Project :L-MNL @File :data_manager.py
@IDE :PyCharm @Author : hejunshu 
@Date :2022/5/18 16:37
"""
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_data(data_ori, unique_catagory, model_name, saqure=None):
    X = []
    Q = []
    scale = 100.0
    choice_0 = data_ori[data_ori['CHOICE'] == 0].index
    CAR_AV_0 = data_ori[data_ori['CAR_AV'] == 0].index
    TRAIN_AV_0 = data_ori[data_ori['TRAIN_AV'] == 0].index
    SM_AV_0 = data_ori[data_ori['SM_AV'] == 0].index

    data_ori.drop(index=choice_0, inplace=True)
    data_ori.drop(index=CAR_AV_0, inplace=True)
    data_ori.drop(index=TRAIN_AV_0, inplace=True)
    data_ori.drop(index=SM_AV_0, inplace=True)

    data_ori['INCOME'].replace(0, 1, inplace=True)
    PURPOSE = data_ori['PURPOSE'].values
    FIRST = data_ori['FIRST'].values
    TICKET = data_ori['TICKET'].values
    WHO = data_ori['WHO'].values
    LUGGAGE = data_ori['LUGGAGE'].values
    AGE = data_ori['AGE'].values
    MALE = data_ori['MALE'].values
    INCOME = data_ori['INCOME'].values
    GA = data_ori['GA'].values
    ORIGIN = data_ori['ORIGIN'].values
    DEST = data_ori['DEST'].values
    SM_SEATS = data_ori['SM_SEATS'].values

    person_ID = data_ori['ID'].values

    unique = [set(PURPOSE), set(FIRST), set(TICKET), set(WHO), set(LUGGAGE), set(AGE),
              set(MALE), set(INCOME), set(GA), set(ORIGIN), set(DEST), set(SM_SEATS)]
    Q_all = np.array([PURPOSE, FIRST, TICKET, WHO, LUGGAGE, AGE, MALE, INCOME, GA, ORIGIN, DEST, SM_SEATS])
    if model_name == 'L_MNL' or model_name == "ASU_DNN" or model_name == "MNL":
        Q_all = (Q_all - Q_all.mean(axis=0)) / (Q_all.std(axis=0))
        saqure = 0
    if (model_name == "E_MNL") or (model_name == "T_MNL") or (model_name == "ET_MNL"):
        Label_list = np.array(range(0, unique_catagory, 1))
        np.random.shuffle(Label_list)
        x_length = 0
        # un = 0
        for m in unique:
            # un += len(m)
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

    for person_id in set(person_ID):
        data = data_ori[data_ori['ID'] == person_id]

        CHOICE = data['CHOICE'].values
        CHOICE_Car = (CHOICE == 3)
        CHOICE_SM = (CHOICE == 2)
        CHOICE_Train = (CHOICE == 1)

        TT_Car = data['CAR_TT'].values / scale
        TT_SM = data['SM_TT'].values / scale
        TT_Train = data['TRAIN_TT'].values / scale

        Cost_Car = data['CAR_CO'].values / scale
        Cost_SM = data['SM_CO'].values * (data['GA'].values == 0) / scale
        Cost_Train = data['TRAIN_CO'].values * (data['GA'].values == 0) / scale

        He_SM = data['SM_HE'].values / scale
        He_Train = data['TRAIN_HE'].values / scale

        ASCs = np.ones(CHOICE.size)
        ZEROs = np.zeros(CHOICE.size)

        X_house = np.array(
            [[ZEROs, ZEROs, TT_Train, Cost_Train, He_Train, CHOICE_Train],
             [ZEROs, ASCs, TT_SM, Cost_SM, He_SM, CHOICE_SM],
             [ASCs, ZEROs, TT_Car, Cost_Car, ZEROs, CHOICE_Car]])
        X_house = np.swapaxes(X_house, 0, 2)

        target_col_index = np.where(person_ID == person_id)[0]
        Q_house = Q_all[:, target_col_index]
        Q_house = np.swapaxes(Q_house, 0, 1)
        X.append(X_house)
        Q.append(Q_house)
    return X, Q, saqure


def load_data(unique_catagory, model_name):
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = pd.read_csv(filePath + 'data_process.csv')
    main_data, extra_data, saqure = process_data(data, unique_catagory, model_name)
    return main_data, extra_data, saqure

def load_data_totest(unique_catagory,model_name, saqure):
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = pd.read_csv(filePath + 'data_process.csv')
    main_data, extra_data, _ = process_data(data, unique_catagory, model_name, saqure)
    return main_data, extra_data

def swiss_load_client():
    filePath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/data' + '/'
    data = np.loadtxt(filePath + 'swissmetor.dat', skiprows=1)  # new是重新切分的

    # 集中处理数据

    CHOICE = data[:, -1]
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

    # main_data = np.array(
    #     [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, GA, AGE, ZEROs, ZEROs, CHOICE_TRAIN],
    #      [ZEROs, ASCs, SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, GA, ZEROs, ZEROs, SM_SEATS, CHOICE_SM],
    #      [ASCs, ZEROs, CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, ZEROs, ZEROs, LUGGAGE, ZEROs, CHOICE_CAR]])
    # 前两项用于构建ASC ，TT, COST, HE, GA, AGE, LUGGAGE, SEATS, CHOICE 输入线性模型 X1 7+2(ASC)个特征输入线性
    #
    # if simpleArchitecture:
    main_data = np.array(
        [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
         [ZEROs, ASCs, SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, CHOICE_SM],
         [ASCs, ZEROs, CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, CHOICE_CAR]])
    # 前两项用于构建ASC ，TT, COST, HE, CHOICE 输入线性模型 X2 7个特征输入非线性

    # if lmnlArchitecture:
    #     main_data = np.array(
    #         [[TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
    #          [SM_TT_SCALED, SM_COST_SCALED, SM_HE_SCALED, CHOICE_SM],
    #          [CAR_TT_SCALED, CAR_CO_SCALED, ZEROs, CHOICE_CAR]])
    # TT, COST, HE, CHOICE 输入线性模型 没有ASC X2 3个特征输入线性

    main_data = np.swapaxes(np.swapaxes(np.swapaxes(main_data, 0, 2), 1, 3), 2, 3)  # (people, menu, beta, choice)

    # if simpleArchitecture or lmnlArchitecture:
    # Hybrid Simple
    extra_data = np.delete(total_data, [18, 19, 21, 22, 25, 26, 27, 20, 23, 0, 1, 2, 3, 15, 16, 17], axis=2)
    # if simpleArchitecture:
    extra_data[:, :, 7][extra_data[:, :, 7] == 0] = 1
    # 18,19,21,22,25,26分别是TT,CO; 20,23是HE X2线性部分的特征；27是CHOICE；0,1,2,3是GROUP,SURVEY,SP,ID；15, 16,17是AV
    # 保留12个特征
    # else:
    #     # Hybrid MNL
    #     extra_data = np.delete(total_data,
    #                            [18, 19, 21, 22, 25, 26, 27, 20, 23, 0, 1, 2, 3, 8, 9, 12, 24, 15, 16, 17],
    #                            axis=2)
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
