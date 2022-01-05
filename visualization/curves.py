import pandas as pd
import re
import numpy as np
import argparse

import matplotlib.pyplot as plt
import copy as cp

def get_args_parser():
    parser = argparse.ArgumentParser('Curves', add_help=False)
    parser.add_argument('--which_curve', default='loss', type=str,
                        choices=['loss', 'acc'],
                        help="""Chose a curve to draw""")
    parser.add_argument('--logFileList', nargs='+', type=str, default='./log.txt')
    parser.add_argument('--accFileList', nargs='+', type=str, default='./student_knn_acc.txt')
    parser.add_argument('--experimentList', nargs='+', type=str, default='current experiment')
    parser.add_argument('--outpath', type=str, default='./')
    return parser

# Example line of log
# {"train_loss": 0.27636545452314193, "train_lr": 4.639659460134639e-06, "train_wd": 0.35833078153189346, "epoch": 389}
def loadLog(logFile):
    df = pd.DataFrame(columns=['epoch', 'loss'])
    f = open(logFile)
    line = f.readline()
    while line:
        numbers = np.array(
            re.findall("\{\"train_loss\": (.+?), \"train_lr\": (.+?), \"train_wd\": (.+?), \"epoch\": (.+?)\}",line)).astype(
            np.float)
        numbers = numbers[0]
        aline={'epoch': numbers[3], 'loss' : numbers[0]}
        df = df.append(aline, ignore_index=True)
        line = f.readline()
    f.close()
    return df

# epoch: 0, 10-NN classifier result: Top1: 10.774246128769356, Top5: 57.815810920945395
def loadAcc(accFile):
    df = pd.DataFrame(columns=['epoch', 'acc'])
    f = open(accFile)
    line = f.readline()
    while line:
        numbers = np.array(
            re.findall("epoch: (.+?), 10-NN classifier result: Top1: (.+?), Top5: (.+?)",line)).astype(
            np.float)
        numbers = numbers[0]
        aline={'epoch': numbers[0], 'acc' : numbers[1]}
        df = df.append(aline, ignore_index=True)
        line = f.readline()
    f.close()
    return df

def drawCurve(df, y_columns,args):
    df.plot(x='epoch', y=y_columns)
    plt.savefig(args.outpath, dpi=100)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Curves', parents=[get_args_parser()])
    args = parser.parse_args()
    df = None
    if args.which_curve == "loss":
        title = 'loss'
        lossdf = pd.DataFrame(columns=['epoch'])
        for fileName in args.logFileList:
            df = loadLog(fileName)
            lossdf = pd.merge(left=lossdf, right=df, left_on='epoch', right_on='epoch', how='outer')
        if type(args.experimentList) == str:
            drawCurve(lossdf, 'epoch', 'loss', args)
        else:
            columnList = cp.deepcopy(args.experimentList)
            columnList.insert(0, 'epoch')
            lossdf.columns = columnList
            drawCurve(lossdf, args.experimentList, args)
    elif args.which_curve == "acc":
        print('acc')
        accdf = pd.DataFrame(columns=['epoch'])
        for fileName in args.accFileList:
            df = loadAcc(fileName)
            accdf = pd.merge(left=accdf, right=df, left_on='epoch', right_on='epoch', how='outer')
        if type(args.experimentList) == str:
            drawCurve(lossdf, 'epoch', 'acc', args)
        else:
            columnList = cp.deepcopy(args.experimentList)
            columnList.insert(0, 'epoch')
            accdf.columns = columnList
            drawCurve(accdf, args.experimentList, args)