import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import math
import collections

data_path = './data/IP/DHCP.csv'
data = []
idx_half = 0
figure_ip_path = './figure_ip'

def readData():
    global data_path
    global data
    global idx_half
    
    data = pd.read_csv(data_path)    
    idx_half = data.index[data['Timestamp'] == '20210630_2350-0000'].tolist()[0]

def statisticalAnalysis(saveHistogram, saveFig, saveSepData), :
    global data
    global figure_ip_path

    if (saveHistogram or saveFig) and not os.path.isdir(figure_ip_path):
        os.mkdir(figure_ip_path)

    sep_data = {}    
    for data_domain in data.keys():
        if 'Timestamp' != data_domain and 'Predict' != data_domain:
            y = np.array(data[data_domain])
            y = y[np.isfinite(y)]
            if saveFig:    
                plt.title(data_domain)
                plt.plot(y)
                save_path = os.path.join(figure_ip_path, data_domain + '_data.png')
                plt.savefig(save_path)
                plt.clf()

            if saveHistogram:
                plt.title(data_domain)
                plt.hist(y)
                plt.yscale('log')
                savePath = os.path.join(figure_ip_path, data_domain + '_histogram.png')
                plt.savefig(savePath)
                plt.clf()

            if saveSepData:
                sep_data[data_domain] = []
                for (i, d) in enumerate(data[data_domain]):
                    if i >= idx_half + 1:
                        sep_data[data_domain].append(d)
                
    if saveSepData:
        sep_data = pd.DataFrame(sep_data)
        sep_data.to_csv('./DHCP_SEP.csv', index= None)

def RuleBasedPrediction():
    global data
    global idx_half

    data_legnth = len(data['Timestamp'])
    prediction = []

    # Detect ssr, sse above the threshold.
    # The threshold is determined by the Histogram.
    # Histogram will be replaced by KDE (Kernel Density Estimate)
    for i in tqdm(range(0, data_legnth)):
        isAttack = 0
        ss_r = data['Ss_request'][i]
        if not math.isnan(ss_r):
            if int(ss_r) >= 13:
                isAttack = 1
        ss_e = data['Ss_Established'][i]
        if not math.isnan(ss_e):
            if ss_e >= 85:
                isAttack = 1
        if i >= idx_half + 1:
            prediction.append(isAttack)
    
    sep_data = pd.read_csv('./backup/IP_answer_dd.csv')
    sep_sse = sep_data['Ss_Established']
    sep_ssr = sep_data['Ss_request']
    
    # Grouping detected data.
    # When the attacker takes over network (DoS or DDosS) or spoofing packet, the outliers' distribution are grouped.
    attack_group = []
    is_start_attack = 0
    non_attack_Count = 0
    for i in range(0, len(prediction)):
        if prediction[i] == 1 and is_start_attack == 0:
            is_start_attack = 1
            attack_group.append([i])
        
        if is_start_attack == 1:
            if prediction[i] == 0:
                non_attack_Count += 1
            else:
                non_attack_Count = 0

        if non_attack_Count == 2:
            is_start_attack = 0
            non_attack_Count = 0
            attack_group[-1].append(i - 2)
    
    # For each group, check the top and bottom 5 data.
    # An outlier may not occur immediately after an attack.
    for ag in attack_group:
        before = ag[0]
        for i in range(0, 5):
            s = sep_sse[ag[0] - i]
            if not math.isnan(s):
                before -= 1
            else:
                break
        after = ag[1]
        for i in range(0, 5):
            s = sep_sse[ag[1] - i]
            if not math.isnan(s):
                after += 1
            else:
                break
        
        # Find the maximum value of sse.
        max_sse = 0
        for i in range(before, after + 1):
            max_sse = max(max_sse, sep_sse[i])

        # Individual rules are applied according to the maximum value of sse.
        # Thresholds of the boundary change according to the attack pattern.
        # Rules are usually set based on specifications, but it is also possible through reverse engineering (inferences based on data).
        if max_sse >= 160:
            for i in range(before, after + 1):
                if sep_sse[i] > 20:
                    prediction[i] = 1
        elif max_sse >= 120:
            for i in range(before, after + 1):
                if sep_sse[i] > 50:
                    prediction[i] = 1
        elif max_sse >= 110:
            for i in range(before, after + 1):
                if sep_sse[i] > 45:
                    prediction[i] = 1
        elif max_sse >= 105:
            for i in range(before, after + 1):
                if sep_sse[i] > 45:
                    prediction[i] = 1
        elif max_sse >= 100:
            for i in range(before, after + 1):
                if sep_sse[i] > 55 and sep_ssr[i] != 0 and sep_sse[i] / sep_ssr[i] > 12:
                    prediction[i] = 1
        elif max_sse >= 95:
            for i in range(before, after + 1):
                if sep_sse[i] >= 45:
                    prediction[i] = 1
        elif max_sse >= 90:
            for i in range(before, after + 1):
                if sep_sse[i] < 89:
                    prediction[i] = 0
        elif max_sse >= 87:
            for i in range(before, after + 1):
                if sep_sse[i] > 40:
                    prediction[i] = 1
        elif max_sse >= 85:
            for i in range(before, after + 1):
                if sep_sse[i] > 50:
                    prediction[i] = 1

    print(collections.Counter(prediction))
    prediction = pd.DataFrame(prediction, columns=['Prediction'])
    print(f'예측 결과. \n{prediction}\n')
    prediction.to_csv('./IP_Rule_Based_Result.csv')

if __name__ == '__main__':
    # Load Data
    readData()

    # Data analysis: Calculate Statistic
    try:
        # need DHCP_SEP.csv
        sep_data = pd.read_csv('./DHCP_SEP.csv')
    except:
        # generate DHCP_SEP.csv
        statisticalAnalysis(True, True, True)

    # Rule Based Detection
    RuleBasedPrediction()

