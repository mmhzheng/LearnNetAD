import numpy as np


def parseDataFile(fname):
    # 从绝对路径获得文件名 003_UCR_Anomaly_DISTORTED3sddb40tr_35000_1st_46600.csv
    fname = fname.split('/')[-1]
    fnamelist = fname.split('_')
    prefix = '_'.join(fnamelist[:-3])
    test_loc, ab_l = int(fnamelist[-3]),int(fnamelist[-1][:-4])
    return prefix, test_loc, ab_l

def range_convers_new(label):
    '''
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
    anomaly_ends,  = np.where(np.diff(label) == -1)
    if len(anomaly_ends):
        if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
            # we started with an anomaly, so the start of the first anomaly is the start of the labels
            anomaly_starts = np.concatenate([[0], anomaly_starts])
    if len(anomaly_starts):
        if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
            # we ended on an anomaly, so the end of the last anomaly is the end of the labels
            anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
    return list(zip(anomaly_starts, anomaly_ends))