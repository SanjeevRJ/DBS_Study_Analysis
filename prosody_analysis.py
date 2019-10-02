import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wav7 = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\7_wav.csv")
wav7 = wav7.drop(["Unnamed: 0"], 1)
wav8 = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\8_wav.csv")
wav8 = wav8.drop(["Unnamed: 0"], 1)
wav9 = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\9_wav.csv")
wav9 = wav9.drop(["Unnamed: 0"], 1)
wav10 = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\10_wav.csv")
wav10 = wav10.drop(["Unnamed: 0"], 1)
wav11 = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\11_wav.csv")
wav11 = wav11.drop(["Unnamed: 0"], 1)

gemaps = pd.read_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\gemaps_features_2019-05-07-21-10-14.csv")
gemaps = gemaps.drop(["Unnamed: 0"], 1)

def gemapsPatient(minRow, maxRow):
    patient = []
    for i in range(minRow, maxRow + 1):
        patient.append(gemaps.iloc[[i]])
    df_patient = pd.concat(patient, axis = 0, sort = True)
    df_patient = df_patient.reset_index()
    df_patient = df_patient.drop(['index'], 1)
    return df_patient

gemaps10 = gemapsPatient(0, 9)
gemaps11 = gemapsPatient(10, 19)
gemaps7 = gemapsPatient(20, 29)
gemaps8 = gemapsPatient(30, 39)
gemaps9 = gemapsPatient(40, 49)

gemaps = [gemaps7, gemaps8, gemaps9, gemaps10, gemaps11]
df_gemaps = pd.concat(gemaps, axis = 0, sort = True)
df_gemaps = df_gemaps.reset_index()
df_gemaps = df_gemaps.drop(['index'], 1)

df_num_gemaps = df_gemaps.drop(['AudioFile'], 1)

wavs = [wav7, wav8, wav9, wav10, wav11]
df_wav = pd.concat(wavs, axis = 0, sort = True)
df_wav = df_wav.reset_index()
df_wav = df_wav.drop(['index'], 1)
#df_wav.to_csv(r"C:\Users\sanje\Desktop\DBS_Prosody_features\wav.csv", index = None, header = True)

df_num_wav = df_wav.drop(['AudioFile'], 1)

Motor_Scores = {"patient_id" : [7 , 8 , 9 , 10 , 11], "on" : [30, 15, 21, 24, 40],
                "off" : [36, 23, 27, 45, 51]}
df_ms = pd.DataFrame(data = Motor_Scores)


Recording_Keys = ["ahh", "sentence", "freespeech", "counting", "animal"]

Recordings = {"sentence": [r"_sentence_off.wav", r"_sentence_on.wav"],
            "freespeech": [r"_freespeech_off.wav", r"_freespeech_on.wav"],
            "ahh": [r"_ahh_off.wav", r"_ahh_on.wav"],
            "counting": [r"_counting_off.wav",  r"_counting_on.wav"],
            "animal": [r"_animal_off.wav", "_animal_on.wav"]}
df_recordings = pd.DataFrame(data = Recordings)

Features = [r"Mean_Pause_Length_VADInt_", r"Pause_Percentage_VADInt_",
            r"Pause_Speech_Ratio_VADInt_", r"Pause_Time_VADInt_",
            r"Pause_Variability_VADInt_", r"Speech_Time_VADInt_",
            r"Total_Time_VADInt_"]

def getValue(wav, fileNum, recording, feature):
    df = pd.DataFrame(wav)
    row = df.loc[df["AudioFile"] == str(fileNum) + recording]
    value = row[feature]
    return value

def makeDataFrame(recording, feature):
    fileNums = [7, 8, 9, 10, 11]
    wavs = [wav7, wav8, wav9, wav10, wav11]
    data = [];
    for i in range(5):
        data.append(getValue(wavs[i], fileNums[i], recording, feature))
    series = pd.concat(data)
    df = pd.DataFrame(series)
    df = df.reset_index()
    df = df.drop(['index'], 1)
    return df

def compareTwoDFs(recording, feature):
    on = makeDataFrame(df_recordings.at[1, recording], feature)
    on.columns = [feature + "on"]
    off = makeDataFrame(df_recordings.at[0, recording], feature)
    off.columns = [feature + "off"]
    frames = [on, off]
    joint = pd.concat(frames, axis = 1, sort = True)
    return joint

def makeDataFrameAllR(feature, DBS_state):
    state = -1
    if DBS_state.find("off") >= 0:
        state = 0
    if DBS_state.find("on") >=0:
        state = 1
    fileNums = [7, 8, 9, 10, 11]
    wavs = [wav7, wav8, wav9, wav10, wav11]
    data = [];
    for i in range(len(wavs)):
        for j in range(len(Recordings)):
            data.append(getValue(wavs[i], fileNums[i], 
                                 Recordings[Recording_Keys[j]][state], feature))
    series = pd.concat(data)
    df = pd.DataFrame(series)
    df = df.reset_index()
    df = df.drop(['index'], 1)
    return df

def makeDataFrameAllRGM(DBS_state, feature):
    col = -1
    for curCol in range(len(df_num_gemaps.columns)):
        curFeat = df_num_gemaps.columns[curCol]
        if curFeat == feature:
            col = curCol 
    state = -1
    if DBS_state.find("off") >= 0:
        state = 0
    if DBS_state.find("on") >=0:
        state = 1  
    data = []
    for ind in range(len(df_gemaps.index)):
        if state == 0 and ind % 2 == state:
            data.append(df_num_gemaps.iat[ind, col])
        if state == 1 and ind % 2 == state:
            data.append(df_num_gemaps.iat[ind, col])
    return data
 
#Performs a paired T-test on on and off data for all recordings for all prosody features
def ttestAllRPros():
    ttest_res = dict()
    for feat in range(len(Features)):
        curFeature = Features[feat] + str(3)
        df_off = makeDataFrameAllR(Features[feat] + "3", "off")
        df_on = makeDataFrameAllR(Features[feat] + "3", "on")
        ttest = ttest_rel(df_off, df_on)
        if ttest_res.get(curFeature) == None:
            ttest_list = [ttest.pvalue, ttest.statistic]
            ttest_res.update({curFeature : ttest_list})
        else:
            curList = ttest_res[curFeature]
            curList.append([ttest.pvalue, ttest.statistic])
            ttest_res[curFeature] = curList
    df_ttest = pd.DataFrame(ttest_res)
    return df_ttest

def ttestAllRGeM(df):
    ttest_res = dict()
    for col in range(len(df_num_gemaps.columns)):
        curFeature = df_num_gemaps.columns[col]
        off = []
        on = []
        for ind in range(len(df_num_gemaps.index)):
            if ind % 2 == 0:
                off.append(df.iat[ind, col])
            else: 
                on.append(df.iat[ind, col])
        ttest = ttest_rel(off, on)
        if ttest_res.get(curFeature) == None:
            ttest_list = [ttest.pvalue, ttest.statistic]
            ttest_res.update({curFeature : ttest_list})
        else:
            curList = ttest_res[curFeature]
            curList.append([ttest.pvalue, ttest.statistic])
            ttest_res[curFeature] = curList
    df_ttest = pd.DataFrame(ttest_res)
    return df_ttest
ttestGM = ttestAllRGeM(df_num_gemaps)

def sigttestAllGM(df_ttest, alpha):
    sigFeat = df_ttest
    for i in range(len(df_ttest.columns)):
        cur_pvalue = df_ttest.iat[0, i]
        if cur_pvalue >= alpha:
            key = df_ttest.columns[i]
            sigFeat = sigFeat.drop(columns=[key])
    #df_sf = pd.DataFrame(sigFeat)
    #return df_sf
    return sigFeat
sigttestGM = sigttestAllGM(ttestGM, 0.1)
#sigttestGM = sigttestGM.rename(index={0: 'pvalue'})
#sigttestGM = sigttestGM.rename(index={1: 't statistic'})

def sigMeanVarGM(df_sig):
    mv_dict = dict()
    for col in range(len(df_sig.columns)):
        curCol = df_sig.columns[col]
        off = pd.DataFrame(makeDataFrameAllRGM('off', curCol))
        on = pd.DataFrame(makeDataFrameAllRGM('on', curCol))
        off_mean = off.mean().get_value(0, 0)
        on_mean = on.mean().get_value(0, 0)
        off_std = off.std().get_value(0, 0)
        on_std = on.std().get_value(0, 0)
        if mv_dict.get(curCol) == None:
            mv_list = [off_mean, off_std, on_mean, on_std]
            mv_dict.update({curCol : mv_list})
        else:
            curList = mv_dict[curCol]
            curList.append([off_mean, off_std, on_mean, on_std])
            mv_dict[curCol] = curList
    df_mv = pd.DataFrame(mv_dict)
    return df_mv
sig_mean_var_gm = sigMeanVarGM(sigttestGM)

def heapMap(df, fileName):
    size = len(df.columns)
    f,ax = plt.subplots(figsize=(size, size))
    sns_plot = sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    fig = sns_plot.get_figure()
    fig.savefig(r"C:\Users\sanje\Desktop\DBS_Prosody_features\Figures\\" + fileName +
                ".png", bbox_inches = 'tight')
    #plt.show()

def fileNum(string):
    for i in range(7, 12):
        if string.find(str(i)) >= 0:
            return i

new_ms = list()
for i in range(len(df_wav)):
    curFile = df_wav.at[i, 'AudioFile'];
    if curFile.find("on") >= 0:
        num = fileNum(curFile)
        ms = df_ms.at[num - 7, 'on']
        new_ms.append(ms)
    else:
        num = fileNum(curFile)
        ms = df_ms.at[num - 7, 'off']
        new_ms.append(ms)

df_ms_val = pd.DataFrame(new_ms)
df_old_wav = df_wav
df_wav['ms'] = new_ms

df_wav_int3 = df_wav
for i in range(len(df_wav.columns)):
    curColumn = df_wav.columns[i]
    if curColumn.find("1") >= 0 or curColumn.find("2") >= 0:
        df_wav_int3 = df_wav_int3.drop([df_wav.columns[i]], 1)
df_wav_int3.columns = ['AudioFile', 'Mean Pause Length', 'Pause Percentage', 'Pause Speech Ratio',
                       'Pause Time', 'Pause Variability', 'Speech Time', 'Total Time', 
                       'Motor Scores']

#corr = df_old_wav.corrwith(df_wav['ms'])

#taken from visualization.py
def nonSquareHM(df1, df_col, fileName):
    corr = df1.corrwith(df_col)
    df_corr = pd.DataFrame(corr)
    size1 = len(df1.columns)
    size2 = len(df_col)
    f,ax = plt.subplots(figsize=(size1, size2))
    sns_plot = sns.heatmap(df_corr, annot=True, linewidths=.5, fmt= '.1f', ax=ax)
    fig = sns_plot.get_figure()
    #plt.show()
    fig.savefig(r"C:\Users\sanje\Desktop\DBS_Prosody_features\Figures\\" + 
                     fileName)

def sigGMCorr(sig):
    gemaps_ms_corr = df_num_gemaps.corrwith(df_wav['ms'])
    sig_gemaps_corr = []
    for i in range(len(gemaps_ms_corr)):
        if abs(gemaps_ms_corr[i]) > sig:
            sig_gemaps_corr.append([gemaps_ms_corr.index[i], 
                                    gemaps_ms_corr[i]])
    return sig_gemaps_corr

sigGMCorrs = sigGMCorr(0.6)

def sigWavCorr(sig):
    gemaps_ms_corr = df_num_wav.corrwith(df_wav['ms'])
    sig_gemaps_corr = []
    for i in range(len(gemaps_ms_corr)):
        if abs(gemaps_ms_corr[i]) > sig:
            sig_gemaps_corr.append([gemaps_ms_corr.index[i], 
                                    gemaps_ms_corr[i]])
    return sig_gemaps_corr

sigWavCorrs = sigWavCorr(0.5)

def transposeDF(df):
    df = df.transpose()
    df = df.reset_index()
    df = df.drop(['index'], 1)
    df_columns = df.iloc[[0]]
    df.columns= df_columns.values.tolist()
    df = df.drop(0)
    df = df.apply(pd.to_numeric)
    return df

def DBS_HM(df, DBS_state, fileName):
    for i in range(df.count()[0]):
        curFile = df.at[i, 'AudioFile'];
        if curFile.find(DBS_state) >= 0:
            df = df.drop(i)
    df = transposeDF(df)
    size = len(df.columns)
    f,ax = plt.subplots(figsize=(size, size))
    sns_plot = sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax, 
                vmin = 0.5, vmax = 1)
    fig = sns_plot.get_figure()
    fig.savefig(r"C:\Users\sanje\Desktop\DBS_Prosody_features\Figures\\" + 
                     fileName)

def all_DBS_HM(dfs, name):
    for i in range(len(dfs)):
        num = str(i + 7)
        DBS_HM(dfs[i], "on", + name + num + "_off.png")
        DBS_HM(dfs[i], "off", + name + num + "_on.png")

#def multHM(dfs, DBS_State):
     
#taken from feature_transformation.py
def pca_plot(df, fileName):
    df_pca = PCA().fit(df);
    explained_var = np.cumsum(df_pca.explained_variance_ratio_)
    plt.plot(explained_var)
    plt.grid(True)
    plt.title('(PCA) Num. of Components vs. Cumulative explained variance')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig(r"C:\Users\sanje\Desktop\DBS_Prosody_features\Figures" + 
                fileName)

    


