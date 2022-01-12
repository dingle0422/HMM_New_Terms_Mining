from collections import defaultdict
import pickle
import numpy as np
import configparser
import os

# 初始化一些数据
os.system('python corpus_prep.py')
conf = configparser.ConfigParser()
conf.read(r"./config.txt")
corp = pickle.load(open(r"./data/{}".format(conf.get("data_name","corp_file")), 'rb'))
w_att = set([o[1] for i in corp for o in i]) # 所有词性种类 -> set


# 头部隐层（词性）概率
def head_mat(corp):
    '''将已经标号BMES词性的语料进行3类矩阵的构建
    corp: BMES语料 -> list of list
    '''
    h_head = defaultdict(int)
    for s in corp:
        for w in s:
            h_head[w[1]] += 1 # BMES frequency
            
    allc = sum(h_head.values()) # total word num

    for k,v in h_head.items():
        h_head[k] = np.log(v/allc) # 将概率指数化，便于计算 log(a) + log(b) = log(a*b)

    return h_head

def trans_mat(corp):
    '''隐藏层转换概率
    corp: BMES语料 -> list of list
    '''
    h_trans = {i:defaultdict(int) for i in w_att} # 初始化一个 -> dict of dict

    tot_trans = defaultdict(int) # 保存每一种t-1词性转换的次数
    for s in corp: #记录每个语料中，每个隐层的转换频率
        for i in range(len(s)-1):
            h_trans[s[i][1]][s[i+1][1]] += 1
            tot_trans[s[i][1]] += 1

    for k in h_trans.keys(): # St-1词性
        for kk in ['s','sn','b','e','m','c']: # St词性
            if kk in h_trans[k].keys(): # 如果存在某两个词性的转换，则通过
                h_trans[k][kk] = np.log(h_trans[k][kk]/tot_trans[k])
            else: # 否则给负无穷，表示概率为0
                h_trans[k][kk] = -np.inf

    return h_trans

def emit_mat(corp):
    '''显层转换概率
    corp: BMES语料 -> list of list
    '''
    # 显层发射概率
    h_launch = {}
    for a in w_att: # BEMS等
        sub = defaultdict(int) # k:v -> word：freq
        att_launch = 0 # 记录该词性一共emit了几次
        for s in corp:
            for i in range(len(s)):
                if s[i][1] == a: # 若词性为a
                    sub[s[i][0]] += 1 # k:v -> word：freq
                    att_launch += 1
                            
        for k,v in sub.items(): #word: freq
            if v > 0: 
                sub[k] = np.log(v/att_launch)
            else:
                sub[k] = -np.inf
            
        h_launch[a] = sub

    return h_launch


if __name__ == "__main__":
    t = trans_mat(corp)
    e = emit_mat(corp)
    for n,m in zip(("t","e"),(t,e)):
        pickle.dump(m, open(r"./data/{}_mat.pkl".format(n), "wb"))

    print(t)