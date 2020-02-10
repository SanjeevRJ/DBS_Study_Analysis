import pandas as pd
from scipy.stats import ttest_rel

CompF7 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\lingComplex_features_7.csv")
CompF8 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\lingComplex_features_8.csv")
CompF9 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\lingComplex_features_9.csv")
CompF10 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\lingComplex_features_10.csv")
CompF11 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\lingComplex_features_11.csv")

comps = [CompF7, CompF8, CompF9, CompF10, CompF11]
df_comps = pd.concat(comps, axis = 0, sort = True)
df_comps = df_comps.reset_index()
df_comps = df_comps.drop(['index'], 1)
df_comps = df_comps.drop(["Unnamed: 0"], 1)

CohF7 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\coherence_features_7.csv")
CohF8 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\coherence_features_8.csv")
CohF9 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\coherence_features_9.csv")
CohF10 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\coherence_features_10.csv")
CohF11 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\coherence_features_11.csv")

nltk7 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\nltk_features_7.csv")
nltk7 = nltk7.drop(4)
nltk8 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\nltk_features_8.csv")
nltk9 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\nltk_features_9.csv")
nltk10 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\nltk_features_10.csv")
nltk11 = pd.read_csv(r"C:\Users\sanje\Voice-Analysis-Pipeline\Output_Folder\Language\nltk_features_11.csv")

nltks = [nltk7, nltk8, nltk9, nltk10, nltk11]
df_nltks = pd.concat(nltks, axis = 0, sort = True)
df_nltks = df_nltks.reset_index()
df_nltks = df_nltks.drop(['index'], 1)
df_nltks = df_nltks.drop(["Unnamed: 0"], 1)

Recording_Keys = ["freespeech", "animal"]

Recordings = {"freespeech": [r"_freespeech_off.txt", r"_freespeech_on.txt"],
            "animal": [r"_animal_off.txt", "_animal_on.txt"]}

transCols = {"NLTK" : "Transcipt_File", "COH" : "Transcript", "COMP" : "Transcript"}

def getValue(df, fileNum, transCol, recording, feature, nltk):
    fullRecord = ""
    if not nltk:
        fullRecord = r"C:\Users\sanje\Desktop\NLTK\Text" + str(fileNum) + "\\" +  str(fileNum) + recording
    else:
        fullRecord = str(fileNum) + recording
    row = df.loc[df[transCol] == fullRecord]
    value = row[feature]
    return value

#def getValueComp(df, )

def makeDataFrameAllR(dfs, transCol, feature, DBS_state, nltk):
    state = -1
    if DBS_state.find("off") >= 0:
        state = 0
    if DBS_state.find("on") >=0:
        state = 1
    fileNums = [7, 8, 9, 10, 11]
    data = []
    for i in range(len(fileNums)):
        for j in range(len(Recordings)):
            data.append(getValue(dfs, fileNums[i], transCol,
                                 Recordings[Recording_Keys[j]][state], feature,
                                 nltk))
    series = pd.concat(data)
    df = pd.DataFrame(series)
    df = df.reset_index()
    df = df.drop(['index'], 1)
    return df
 
#Performs a paired T-test on on and off data for all recordings for a single prosody feature
def ttestSingleF(dfs, transCol, feat, nltk):
    df_off = makeDataFrameAllR(dfs, transCol, feat, "off", nltk)
    df_on = makeDataFrameAllR(dfs, transCol, feat, "on", nltk)
    ttest = ttest_rel(df_off, df_on)
    return ttest

ttest_repeat = ttestSingleF(df_nltks, transCols["NLTK"], "repeat", True)
repeat_on = makeDataFrameAllR(df_nltks, transCols["NLTK"], "repeat", "on", True) 
repeat_on_mean = repeat_on.mean().get_value(0,0)
repeat_on_std = repeat_on.std().get_value(0,0)
repeat_off = makeDataFrameAllR(df_nltks, transCols["NLTK"], "repeat", "off", True) 
repeat_off_mean = repeat_off.mean().get_value(0,0)
repeat_off_std = repeat_off.std().get_value(0,0)

ttest_brunet = ttestSingleF(df_comps, transCols["COMP"], "Brunet Index", False)
brunet_on = makeDataFrameAllR(df_comps, transCols["COMP"], "Brunet Index", "on", False) 
brunet_on_mean = brunet_on.mean().get_value(0,0)
brunet_on_std = brunet_on.std().get_value(0,0)
brunet_off = makeDataFrameAllR(df_comps, transCols["COMP"], "Brunet Index", "off", False) 
brunet_off_mean = brunet_off.mean().get_value(0,0)
brunet_off_std = brunet_off.std().get_value(0,0)

ttest_honore = ttestSingleF(df_comps, transCols["COMP"], "Honore Stat", False)
honore_on = makeDataFrameAllR(df_comps, transCols["COMP"], "Honore Stat", "on", False) 
honore_on_mean = honore_on.mean().get_value(0,0)
honore_on_std = honore_on.std().get_value(0,0)
honore_off = makeDataFrameAllR(df_comps, transCols["COMP"], "Honore Stat", "off", False) 
honore_off_mean = honore_off.mean().get_value(0,0)
honore_off_std= honore_off.std().get_value(0,0)

ttest_ttratio = ttestSingleF(df_comps, transCols["COMP"], "Type Token Ratio", False)
ttratio_on = makeDataFrameAllR(df_comps, transCols["COMP"], "Type Token Ratio", "on", False) 
ttratio_on_mean = ttratio_on.mean().get_value(0,0)
ttratio_on_std = ttratio_on.std().get_value(0,0)
ttratio_off = makeDataFrameAllR(df_comps, transCols["COMP"], "Type Token Ratio", "off", False) 
ttratio_off_mean = ttratio_off.mean().get_value(0,0)
ttratio_off_std = ttratio_off.std().get_value(0,0)

def mean_var():
    mv_dict = dict()
    mv_list_repeat = [repeat_off_mean, repeat_off_std, repeat_on_mean, repeat_on_std]
    mv_dict.update({'repeat' : mv_list_repeat})
    mv_list_brunet = [brunet_off_mean, brunet_off_std, brunet_on_mean, brunet_on_std]
    mv_dict.update({'brunet' : mv_list_brunet})
    mv_list_honore = [honore_off_mean, honore_off_std, honore_on_mean, honore_on_std]
    mv_dict.update({'honore' : mv_list_honore})
    mv_list_ttratio = [ttratio_off_mean, ttratio_off_std, ttratio_on_mean, ttratio_on_std]
    mv_dict.update({'type token' : mv_list_ttratio})
    df_mv = pd.DataFrame(mv_dict)
    return df_mv

def tvalues():
    t_dict = dict()
    t_list_repeat = [ttest_repeat.pvalue, ttest_repeat.statistic]
    t_dict.update({'repeat' : t_list_repeat})
    t_list_brunet = [ttest_brunet.pvalue, ttest_brunet.statistic]
    t_dict.update({'brunet' : t_list_brunet})
    t_list_ttratio = [ttest_ttratio.pvalue, ttest_ttratio.statistic]
    t_dict.update({'ttratio' : t_list_ttratio})
    t_list_honore = [ttest_honore.pvalue, ttest_honore.statistic]
    t_dict.update({'honore' : t_list_honore})
    df_t = pd.DataFrame(t_dict)
    return df_t
    