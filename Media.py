import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import os
import math
import collections

data_path_list = ['./data/Media/Media_INFO.csv', './data/Media/Media_LOGIN.csv', './data/Media/Media_MENU.csv', './data/Media/Media_STREAM.csv']
data_list = []
idx_half = 0
figure_media_path = './figure_media'

def readData():
    global data_path_list
    global data_list
    global idx_half

    for path in data_path_list:
        data = pd.read_csv(path)
        data_list.append(data)
    
    idx_half = data_list[0].index[data_list[0]['Timestamp'] == '20171231_2355-0000'].tolist()[0]

def statisticalAnalysis(saveHistogram, saveFig, saveStd):
    global data_list
    global figure_media_path

    total_data = {
        'INFO': {},
        'LOGIN': {},
        'MENU': {}
    }
    for key in total_data.keys():
        total_data[key] = {
            'Request': [],
            'Success': [],
            'Fail': []
        }

    std_obj = {}

    if (saveHistogram or saveFig) and not os.path.isdir(figure_media_path):
        os.mkdir(figure_media_path)

    for data in data_list:
        for data_domain in data.keys():
            if not 'Timestamp' in data_domain:
                if saveHistogram:
                    for domain_key in total_data.keys():
                        if domain_key in data_domain:
                            for type_key in total_data[domain_key].keys():
                                if type_key in data_domain:
                                    total_data[domain_key][type_key].extend(data[data_domain])
                
                y = np.array(data[data_domain])
                y = y[np.isfinite(y)]
                if saveFig:
                    plt.title(data_domain)
                    plt.plot(y)
                    save_path = os.path.join(figure_media_path, data_domain + '.png')
                    plt.savefig(save_path)
                    plt.clf()
                
                if saveStd:
                    data_avg = np.mean(y)
                    data_std =  np.std(y)
                    if data_std != 0:
                        nlist = []
                        for (i, d) in enumerate(data[data_domain]):
                            if not math.isnan(d):
                                temp = round((d - data_avg)/data_std, 3)
                                nlist.append(temp)
                            else:
                                nlist.append(d)
                        std_obj[data_domain] = nlist
        
    if saveHistogram:
        for domain_key in total_data.keys():
            for type_key in total_data[domain_key].keys():
                target_data = total_data[domain_key][type_key]
                histogram_data = []
                for data in target_data:
                    if not math.isnan(data):
                        histogram_data.append(math.floor(data // 100) * 100)
                save_key = domain_key + '_' + type_key 
                plt.title(save_key)
                plt.hist(histogram_data)
                plt.yscale('log')
                savePath = os.path.join(figure_media_path, save_key + '.png')
                plt.savefig(savePath)
                plt.clf()
    
    if saveStd:
        avg = []
        for i in range(0, len(data_list[0]['Timestamp'])):
            std_sum = 0
            std_count = 0
            for data in data_list:
                for data_domain in data.keys():
                    if data_domain != 'Timestamp':
                        d = data[data_domain][i]
                        if not math.isnan(d):
                            std_sum += d
                            std_count += 1

            if std_count != 0:
                std_avg = round(std_sum / std_count, 4)
                avg.append(std_avg)
            else:
                avg.append(0)
        std_obj['avg'] = avg
        std_obj = pd.DataFrame(std_obj)
        std_obj.to_csv('./all_standard.csv')

def getStreamScore(saveFig):
    import rrcf

    global data_list
    global idx_half
    global figure_media_path

    if saveFig and not os.path.isdir(figure_media_path):
        os.mkdir(figure_media_path)

    for data in data_list:
        for data_domain in data.keys():
            if not 'Timestamp' in data_domain and 'STREAM-03-Session' in data_domain:
                print(data_domain)
                data = data_list[3][data_domain]
                data = np.array(data)
                data = data[idx_half + 1:]
                data = data[np.isfinite(data)]
                
                # Set tree parameters
                num_trees = 40
                shingle_size = 48
                tree_size = 256

                # Create a forest of empty trees
                forest = []
                for _ in range(num_trees):
                    tree = rrcf.RCTree()
                    forest.append(tree)

                # Use the "shingle" generator to create rolling window
                points = rrcf.shingle(data, size=shingle_size)

                # Create a dict to store anomaly score of each point
                avg_codisp = {}

                # For each shingle...
                for index, point in tqdm(enumerate(points)):
                    # For each tree in the forest...
                    for tree in forest:
                        # If tree is above permitted size...
                        if len(tree.leaves) > tree_size:
                            # Drop the oldest point (FIFO)
                            tree.forget_point(index - tree_size)
                        # Insert the new point into the tree
                        tree.insert_point(point, index=index)
                        # Compute codisp on the new point...
                        new_codisp = tree.codisp(index)
                        # And take the average over all trees
                        if not index in avg_codisp:
                            avg_codisp[index] = 0
                        avg_codisp[index] += new_codisp / num_trees
                
                score = avg_codisp.values()
                if saveFig:
                    plt.plot(data)
                    plt.plot(score, 'r')

                    savePath = os.path.join(figure_media_path, f'Total_{data_domain}.png')
                    plt.savefig(savePath)
                    plt.clf()

                    savePath = os.path.join(figure_media_path, f'Score_{data_domain}.png')
                    plt.plot(score, 'r')
                    plt.savefig(savePath)
                    plt.clf()
                
                score = pd.DataFrame(score, columns=['score'])
                savePath = f'./score_{str(num_trees)}_{str(shingle_size)}_{data_domain}.csv'
                score.to_csv(savePath)

def RuleBasedPrediction():
    global data_list
    global idx_half

    data_legnth = len(data_list[0]['Timestamp'])
    prediction = []

    exist_rrfc_score = True
    rrfc_score = []
    rrfc_score_count = 0
    try:
        rrfc_score = pd.read_csv('./score_40_48_STREAM_Session_3.csv')['score']
    except:
        exist_rrfc_score = False

    # Detect feature except stream data above the threshold.
    # Detect stream data above the rrfc score threshold.
    # The threshold is determined by the Histogram.
    # Histogram will be replaced by KDE (Kernel Density Estimate).
    for i in tqdm(range(0, data_legnth)):
        isAttack = 0
        for j in range(1, 2):
            r = data_list[0]['INFO-01-Request'][i]
            s = data_list[0]['INFO-01-Success'][i]
            f = data_list[0]['INFO-01-Fail'][i]
            if not math.isnan(r) and s >= 3000:
                isAttack = 1

        for j in range(1, 6):
            r = data_list[1]['LOGIN-0'+str(j)+'-Request'][i]
            s = data_list[1]['LOGIN-0'+str(j)+'-Success'][i]
            f = data_list[1]['LOGIN-0'+str(j)+'-Fail'][i]
            if not math.isnan(r) and r >= 4000:
                isAttack = 1
            if not math.isnan(s) and s >= 4000:
                isAttack = 1
            if not math.isnan(f) and f >= 500:
                isAttack = 1

        for j in range(1, 5):
            r = data_list[2]['MENU-0'+str(j)+'-Request'][i]
            s = data_list[2]['MENU-0'+str(j)+'-Success'][i]
            f = data_list[2]['MENU-0'+str(j)+'-Fail'][i]
            if not math.isnan(r) and r >= 9000:
                isAttack = 1
            if not math.isnan(s) and s >= 9000:
                isAttack = 1
            if not math.isnan(f) and f >= 200:
                isAttack = 1
        
        if exist_rrfc_score:
            s1 = data_list[3]['STREAM-01-Session'][i]
            s2 = data_list[3]['STREAM-02-Session'][i]
            s3 = data_list[3]['STREAM-03-Session'][i]
            if i >= idx_half + 1:
                if not math.isnan(s3):
                    if rrfc_score_count < len(rrfc_score):
                        if rrfc_score[rrfc_score_count] >= 65:
                            isAttack = 2
                    rrfc_score_count += 1

        if i >= idx_half + 1:
            prediction.append(isAttack)

    all_std = pd.read_csv('./all_std.csv')
    std_avg_list = list(all_std['avg'][idx_half + 1:])
    
    # RRCF cannot determine the last 48 packets, then use std > 4 to detect.
    for i in range(len(prediction) - 48, len(prediction)):
        if int(std_avg_list[i]) >= 4:
            prediction[i] = 1
        else:
            prediction[i] = 0

    # Grouping detected data.
    # For generate each group, consider the successive bottom 5 data.
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

        if non_attack_Count == 5:
            is_start_attack = 0
            non_attack_Count = 0
            attack_group[-1].append(i - 5)
    
    # Seperate gropu with group size.
    attack_group_multi = []
    attack_group_single = []
    for ag in attack_group:
        before = ag[0]
        after = ag[-1]
        if after - before > 1:
            attack_group_multi.append(ag)
        elif len(ag) > 1:
            attack_group_single.append(ag)
    
    # For multi group
    for ag in attack_group_multi:
        before = ag[0]
        after = ag[-1]
        max_avg = 0
        attack_count = 0
        # Get maximum std avg.
        # Get attack count.
        for i in range(before, after + 1):
            max_avg = max(max_avg, std_avg_list[i])
            if prediction[i] == 1:
                    attack_count += 1
        for i in range(before, after + 1):
            rmax = 0
            rmin = 10
            for key in all_std:
                r = all_std[key][i + idx_half + 1]
                if 'Request' in key and not math.isnan(r):
                    rmax = max(rmax, r)
                    rmin = min(rmin, r)
                    
            if round(max_avg) <= 3:
                # if maximum std avg < 3 then false grouping.
                # Not to be attack.
                prediction[i] = 0
                
                # Rebound sigle detect.
                if std_avg_list[i] >= 3:
                    prediction[i] = 1
                    for j in range(i - 2, i + 3):
                        # Check for peripheral data.
                        if std_avg_list[j] >= 2.7:
                            prediction[j] = 1
                elif std_avg_list[i] <= 2.7 and std_avg_list[i] >= 2.0:
                    # Check for distinct rule.
                    if rmax > 8.4 and rmin > 2:
                        prediction[i] = 1
                else:
                    # Check for distinct rule.
                    if rmax > 10:
                        prediction[i] = 1
            else:                    
                # True grouping.
                # To be attack.
                if attack_count >= (after - before + 1) // 2:
                    prediction[i] = 1

    # For signal group
    # Check whether exist an attack in the boundary.
    for ag in attack_group_single:
        if ag[0] == 0 or ag[0] != ag[-1]:
            continue
            
        now = ag[0]
        before = ag[0] - 1
        after = ag[0] + 1
        diff1 = std_avg_list[before] - std_avg_list[now]
        diff2 = std_avg_list[after] - std_avg_list[now]

        for i in range(before, after + 1):
            smax = 0
            smin = 10
            rmax = 0
            rmin = 10
            for key in all_std:
                if 'Timestamp' in key or 'avg' in key:
                    continue
                
                x = all_std[key][i + idx_half + 1]
                if math.isnan(x):
                    continue
                
                if 'STREAM' in key:
                    smax = max(smax, x)
                    smin = min(smin, x)
                else:
                    rmax = max(rmax, x)
                    rmin = min(rmin, x)

            if std_avg_list[now] > 3.5:
                # For high maximal.               
                if diff1 < 0 and diff2 < 0:
                    # For extreme maximal.
                    if std_avg_list[i] >= 3 and smax >= 6:
                       prediction[i] = 1
            else:
                # For low maximal      
                if diff1 < 0 and diff2 < 0:
                    # For extreme maximal.
                    if rmax >= 10:
                        # Check for distinct rule.
                        prediction[i] = 1
                elif diff1 < 0 and diff2 > 0:
                    # Take extreme maximal.
                    prediction[i] = 1

    for (i, p) in enumerate(prediction):
        if p == 2:
            prediction[i] = 1

    print(collections.Counter(prediction))
    prediction = pd.DataFrame(prediction, columns=['Prediction'])
    print(f'예측 결과. \n{prediction}\n')
    prediction.to_csv('./Media_Rule_Based_Result.csv', index=False)

if __name__ == '__main__':
    # Load Data
    readData()

    # Data analysis (Calculate Statistic)
    try:
        # Need ALL_STD.csv
        std_obj = pd.read_csv('./all_standard.csv')
    except:
        # Generate ALL_STD.csv
        statisticalAnalysis(True, True, True)
    try:
        # need RRCF score
        # The rrcf score changes slightly each time it is executed.
        # TODO If you generate new rrfc_score, then re-calculate threshold!!!
        rrfc_score = pd.read_csv('./score_40_48_STREAM_Session_3.csv')
    except:
        # Generate RRCF score
        # Calcuating RRCF score takes about 20 ~ 30 minutes depending on CPU.
        getStreamScore(True)

    # Rule Based Detection
    RuleBasedPrediction()
