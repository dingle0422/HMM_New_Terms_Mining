import jieba.posseg as pseg
import pickle
import configparser
from corpus_prep import add_old_terms
from collections import defaultdict
import numpy as np
import re
import logging
import os
cwd = os.path.split(os.path.realpath(__file__))[0]
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(os.path.join(cwd, r"logs/log.log"),encoding="utf-8")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# 是否使用停用词
cwd = os.path.split(os.path.realpath(__file__))[0]
conf = configparser.ConfigParser()
conf.read(os.path.join(cwd , r"config.txt"))
keep_stopwords = conf.getboolean("strategy","keep_stopwords")
# 加入术语
datapath = os.path.join(cwd,"data")
if conf.get("data_name","old"):
    add_old_terms(filepath= os.path.join(datapath,conf.get("data_name","old"))) # tax_term


# 导入矩阵
e_mat = pickle.load(open(os.path.join(cwd, r"data/e_mat.pkl"), 'rb'))
t_mat = pickle.load(open(os.path.join(cwd, r"data/t_mat.pkl"), 'rb'))


def tokenizer(string: str, keep_stopwords = keep_stopwords) -> list:    
    '''先用jieba对原字符串进行切词
    '''
    with open(os.path.join(cwd,r"data/stopwords_chi.txt"), 'r', encoding= 'utf8') as sw:
        stopwords = [i.strip() for i in sw.readlines()]

    string = ",".join([i for i in re.findall(r'[\u4e00-\u9fa5]+', string)])
    if keep_stopwords:
    # print([i for i in pseg.lcut(string) if i.word not in stopwords])
        return [i for i in pseg.lcut(string)]
    else:
        return [i for i in pseg.lcut(string) if i.word not in stopwords]


def viterbi(string, one_sentence = True) -> list:
    '''利用viterbi算法，求一个字符串的最佳词性标注路径
    string: 字符串
    e_mat: 发射概率矩阵（显层）
    t_mat: 转化概率矩阵（隐层）
    one_sentence: 被调用时，为单个句子或txt文件
    '''
    global e_mat, t_mat
    pairs = tokenizer(string)
    word = [i.word for i in pairs]
    # flag = [i.flag for i in pairs]

    # 发射概率
    x_y_dict = defaultdict()
    for i in range(len(word)):
        x_y_dict[i] = {o:e_mat[o].get(word[i], -np.inf) for o in ['s','sn','b','e','m','c']} # 得到输入字符串中，每个字符的发射概率


    state_num = {"s":0 ,"sn":1, "b":2, "e":3, "m":4, 'c':5}

    # T1记录每个时刻隐状态最大概率，T2记录t时刻是由t-1的哪个隐状态转化过来的
    T1,T2 = np.zeros((len(state_num), len(word))), np.zeros((len(state_num), len(word))) # 横轴state，纵轴observation
    for pos in range(len(word)-1):
        t_0_launch = x_y_dict[pos]
        t_1_launch = x_y_dict[pos + 1] 
        # 因为我们要找的是每后面一步的最优路径，所以我们要从后一步的隐层开始遍历
        for k1,v1 in t_1_launch.items(): # states : log-prob
            max_p = -np.inf # 对比更新最大概率
            prob_l = [] # 保存指向同一state的不同路径概率
            mark_l = [] # 保存概率对应的标记
            for k0,v0 in t_0_launch.items():
                if pos == 0:
                    if max_p < v0 + t_mat[k0].get(k1, -np.inf) + v1:
                        prob_l.append(v0 + t_mat[k0].get(k1, -np.inf) + v1) # 头层发射概率 + 下一层转换概率 + 下一层发射概率
                        mark_l.append(k0)

                else:
                    if max_p < T1[state_num[k0], pos] +t_mat[k0].get(k1, -np.inf) + v1:
                        prob_l.append(T1[state_num[k0],pos] + t_mat[k0].get(k1, -np.inf) + v1) # 前面最优路径概率 + 下一层转换概率 + 下一层发射概率
                        mark_l.append(k0)
                


            # 找到这一步的最优路径，保存概率和位置信息
            if not prob_l:
                T1[state_num[k1],pos + 1] = -np.inf
                T2[state_num[k1],pos + 1] = None
            else:
                T1[state_num[k1],pos + 1] = max(prob_l)
                T2[state_num[k1],pos + 1] = state_num[mark_l[prob_l.index(max(prob_l))]]
        

        
    lastcol = list(T1[:,-1])
    # print(T1,T2)
    # print(lastcol)

    if max(lastcol) == -np.inf:
        if one_sentence:
            logger.info("No sequent path can be found in string: {} \n The words tokenized by jieba will be return".format(string))
            print( "No sequent path can be found in string: {}".format(string))
            return pairs
        else:
            assert max(lastcol) != -np.inf, "No sequent path can be found in string: {}".format(string)

    lastmark = lastcol.index(max(lastcol))
    # print(T2)

    path = []
    def find_path(lastmark, pos):
        if pos == -len(word)-1:
            return
        
        path.insert(0, lastmark)
        lastmark = T2[int(lastmark), pos]
        find_path(lastmark, pos-1)
        
    find_path(lastmark, pos = -1)
    decode = {0:"s", 1:"sn", 2:"b", 3:"e", 4:"m", 5:'c'}
    # print([decode[i] for i in path])
    return [decode[i] for i in path], pairs



def get_terms_vtb(string, one_sentence, single = False):
    '''从标注完的字符串中，找到组合起来的领域新词
    string: 字符串
    single: 是否将未组合的单个名词sn也加入新词组中
    one_sentence: 被调用时，为单个句子或txt文件
    '''

    r = viterbi(string, one_sentence)

    if isinstance(r, list): 
        return r

    pairs = r[1]
    marks = r[0]
    if not marks:
        print("This sentence can not be decomposed: \n",string)
        return
    # print(pairs)
    pairs= [[pairs[i].word, marks[i]] for i in range(len(pairs))]
    # print(pairs)
    # b = False
    # m = False
    new_words = []    
    for i in range(len(pairs)):

        if pairs[i][1] == "b":# and pairs[i-1][1] not in ["b","m"]: # b
            word_parts = []
            word_parts.append(pairs[i][0])
            for ii in range(i+1, len(pairs)):

                if pairs[ii][1] == 'm':
                    word_parts.append(pairs[ii][0])

                elif pairs[ii][1] == 'e':
                    word_parts.append(pairs[ii][0])
                    new_words.append("".join(word_parts))
                    break # 取到结尾就结束，寻找下一个begin
                
                # else: # 如果下一步没有任何其他M\E，则B单独成词
                #     new_words.append("".join(word_parts))
                #     break
                else: # 下一步就断开了，直接结束，寻找下一个begin !!! 原版
                    break

        #     b = True
        #     word_parts.append(pairs[i][0])
        
        # elif pairs[i][1] == "m" and i != 0:
        #     if pairs[i-1][1] in ["b",'m']:
        #         m = True
        #         word_parts.append(pairs[i][0]) # bm / ,me
                
        # elif pairs[i][1] == 'e' and i != 0: 
        #     if pairs[i-1][1] in ["b",'m'] and (b or m): # be / ,me
        #         word_parts.append(pairs[i][0])
        #         new_words.append("".join(word_parts))
            
        #         b = False
        #         m = False
        #         word_parts = []
        
        # else:


    if single: # only filter out the component terms  --- BME/BE/BM
        for i in range(len(pairs)):
            if pairs[i][1] == "sn":
                new_words.append(pairs[i][0])
    

    if not new_words:
        if one_sentence:
            logger.info("There is no new term can be found in string: {} \n The words tokenized by jieba will be return".format(string))
            print( "There is no new term can be found in string: {}".format(string))
            return pairs
        else:
            assert new_words, "There is no new term can be found in string: {}".format(string)


    return list(set(new_words))





if __name__ == "__main__":
    a = get_terms_vtb('企税月,季,报自动生成的数据不正确或获取不到,如何处理', one_sentence= True)
    print(a)

