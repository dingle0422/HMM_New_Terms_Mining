import pickle
import os
from collections import defaultdict
import sys
import configparser
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

cwd = os.path.split(os.path.realpath(__file__))[0]
conf = configparser.ConfigParser()
conf.read(os.path.join(cwd , r"config.txt"))
threshold = conf.getint('strategy', 'threshold')

def overlap_out(new_corp, threshold = threshold, keep_single = True):
    """
    找到有重复的长短词，准备删去存在overlap大于threshold的短词
    new_corp: 新的用于挖掘的语料
    threshold: 最低overlap数量阈值
    keep_single: 是否保留在语料中只出现过一次的新词
    """
    
    flatten = {o[0]:o[1] for o in new_corp}

    # 找到有重复的长短词，准备删去存在overlap大于threshold的短词
    # threshold = 3 # 值越小，越严苛;当重叠的长词个数大于该阈值时，说明该短词不完整，该从新词集中删去
    mv = defaultdict(int)
    single = defaultdict(int)
    for i in flatten.keys():
        for o  in flatten.keys():
            if i in o and len(i) != len(o):
                mv[i] += 1
            if i in o:
                single[i] += 1

    rm = {i[0] for i in mv.items() if i[1] > threshold} # 当重叠的长词个数大于该阈值时，说明该短词不完整，该词被列为removed
    
    if not keep_single: # 是否保留语料中document frequency等于1的新词
        single = {k for k,v in single.items() if v == 1} #!!!!!!!!!!!!!!!!!!!! 目前1比较合适,不写入config
        rm = rm | single


    for i in rm:
        del flatten[i]

    return [[k,v] for k, v in flatten.items()]

    




if __name__ == "__main__":

    cwd = os.path.split(os.path.realpath(__file__))[0]
    nt = pickle.load(open(os.path.join(cwd ,r"Mining/NT.pkl"), 'rb'))

    # 开始计算新词词频
    filepath = os.path.join(cwd , r"data/question_add_hf.pkl")
    corp = pickle.load(open(filepath, 'rb'))['question'].tolist()

    word_freq = defaultdict(int)
    newt = overlap_out(nt)
    print(newt,len(newt))

    # for w in newt:
    #     for s in corp:
    #         if w[0] in s:
    #             word_freq[w[0]] += 1


    # r = sorted(word_freq.items(), key = lambda x: x[1], reverse = True)
    # print(r,len(r))