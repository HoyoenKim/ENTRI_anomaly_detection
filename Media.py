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
        std_obj = pd.DataFrame(std_obj)
        print(std_obj)
        std_obj.to_csv('./ALL_STD.csv')

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

    existScore = True
    score = []
    score_save = []
    score_count = 0
    try:
        score = pd.read_csv('./score_40_48_STREAM_Session_3.csv')['score']
    except:
        existScore = False

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
        
        if existScore:
            s1 = data_list[3]['STREAM-01-Session'][i]
            s2 = data_list[3]['STREAM-02-Session'][i]
            s3 = data_list[3]['STREAM-03-Session'][i]
            if i >= idx_half + 1:
                if not math.isnan(s3):
                    if score_count < len(score):
                        if score[score_count] >= 65:
                            isAttack = 2
                        score_save.append(score[score_count])
                    else:
                        score_save.append(-1)
                    score_count += 1
                else:
                    score_save.append(-1)

        if i >= idx_half + 1:
            prediction.append(isAttack)

    handwork = pd.read_csv('./all_std.csv')
    avg = list(handwork['avg'][idx_half + 1:])

    ret = pd.read_csv('./backup/Media_answer_bb_65.csv')['Prediction']
    
    # 2. rrcf 는 맨 뒤 48개 메시지를 분석하지 못하므로 새로운 rule 로 탐지
    # std > 4 이상
    for i in range(len(prediction) - 48, len(prediction)):
        if int(avg[i]) >= 4:
            prediction[i] = 1
        else:
            prediction[i] = 0

    # 1. Attack Group 화

    attack_group = []
    isStart = 0
    zeroCount = 0
    for i in range(0, len(prediction)):
        if prediction[i] == 1 and isStart == 0:
            isStart = 1
            attack_group.append([i])
        
        if isStart == 1:
            if prediction[i] == 0:
                zeroCount += 1
            else:
                zeroCount = 0

        
        if zeroCount == 5:
            isStart = 0
            zeroCount = 0
            attack_group[-1].append(i - 5)
    
    attack_group_filter = []

    for aa in attack_group:
        if aa[-1] - aa[0] > 1:
            attack_group_filter.append(aa)
    
    print(attack_group_filter)

    for aa in attack_group_filter:
        maxScore = 0
        for i in range(aa[0], aa[-1] + 1):
            #prediction[i] = 1
            maxScore = max(maxScore, avg[i])
        
        if round(maxScore) <= 3:
            for i in range(aa[0], aa[-1] + 1):
                print(i, end=' ')
                print(ret[i], end=' ')
                s = 0
                smin = 10
                for key in handwork:
                    if key != 'Timestamp' and not math.isnan(handwork[key][i + idx_half + 1]):
                        if not 'STREAM' in key and not 'avg' in key:
                            s = max(s, handwork[key][i + idx_half + 1])
                            smin = min(smin, handwork[key][i + idx_half + 1])
                    print(handwork[key][i + idx_half + 1], end=' ')
                print()
                
                s = 0
                for key in handwork:
                    if key != 'Timestamp' and not math.isnan(handwork[key][i + idx_half + 1]):
                        s = max(s, handwork[key][i + idx_half + 1])
                prediction[i] = 0

                if maxScore >= 3:
                    if avg[i] >= 2.9:
                        print(smin)
                        prediction[i] = 1
                    
                    if s >= 10:
                        prediction[i] = 1
                elif maxScore >= 2.5:
                    if avg[i] >= 2.5 and s >= 8:
                        prediction[i] = 1
        else:
            c = 0
            for i in range(aa[0], aa[-1] + 1):
                if prediction[i] == 1:
                    c += 1
                    
            # true grouping
            if c >= (aa[-1] - aa[0] + 1) // 2:
                for i in range(aa[0], aa[-1] + 1):
                    prediction[i] = 1

    print(collections.Counter(prediction))
    
    diff = []
    
    for i in range(0, len(ret)):
        if ret[i] == prediction[i] or prediction[i] == 2:
            diff.append(0)
        else:
            diff.append(1)

    print(collections.Counter(diff))
    print(len(handwork['avg'][idx_half + 1:]), len(score_save), len(prediction), len(ret), len(diff))
    
    answer = {
        'Prediction': prediction,
        'Score': score_save,
        'Ret': ret,
        'Diff': diff,
        'Avg': avg
    }
    #prediction = pd.DataFrame(prediction, columns=['Prediction'])
    #print(f'예측 결과. \n{prediction}\n')
    #prediction.to_csv('./Media_Rule_Based_Result.csv', index=False)
    answer = pd.DataFrame(answer)
    answer.to_csv('./ntc.csv')

if __name__ == '__main__':
    # Load Data
    readData()

    # Data analysis (Calculate Statistic)
    try:
        # Need ALL_STD.csv
        std_obj = pd.read_csv('./ALL_STD.csv')
    except:
        # Generate ALL_STD.csv
        statisticalAnalysis(True, True, True)
        None
    try:
        # need RRCF score
        # The rrcf score changes slightly each time it is executed.
        rrfc_score = pd.read_csv('./score_40_48_STREAM-03-Session_1.csv')
    except:
        # Generate RRCF score
        # Calcuating RRCF score takes about 20 ~ 30 minutes depending on CPU.
        getStreamScore(True)

    # Rule Based Detection
    RuleBasedPrediction()
