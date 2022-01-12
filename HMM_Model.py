import jieba.posseg as pseg
import pickle
from collections import defaultdict
import numpy as np
import re

# 导入矩阵
e_mat = pickle.load(open(r"./data/e_mat.pkl", 'rb'))
t_mat = pickle.load(open(r"./data/t_mat.pkl", 'rb'))

def tokenizer(string: str) -> list:    
    '''先用jieba对原字符串进行切词
    '''
    with open(r"./data/stopwords_chi.txt", 'r', encoding= 'utf8') as sw:
        stopwords = [i.strip() for i in sw.readlines()]

    string = ",".join([i for i in re.findall(r'[\u4e00-\u9fa5]+', string)])

    # print([i for i in pseg.lcut(string) if i.word not in stopwords])
    return [i for i in pseg.lcut(string) if i.word not in stopwords]


def viterbi(string) -> list:
    '''利用viterbi算法，求一个字符串的最佳词性标注路径
    string: 字符串
    e_mat: 发射概率矩阵（显层）
    t_mat: 转化概率矩阵（隐层）
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
        for k1,v1 in t_1_launch.items():
            max_p = -np.inf # 对比更新最大概率
            prob_l = [] # 保存指向同一state的不同路径概率
            mark_l = [] # 保存概率对应的标记
            for k0,v0 in t_0_launch.items():
                if pos == 0:
                    if max_p < v0 + t_mat[k0][k1] + v1:
                        prob_l.append(v0 + t_mat[k0][k1] + v1) # 头层发射概率 + 下一层转换概率 + 下一层发射概率
                        mark_l.append(k0)
                else:
                    if max_p < T1[state_num[k0], pos] +t_mat[k0][k1] + v1:
                        prob_l.append(T1[state_num[k0], pos] + t_mat[k0][k1] + v1) 
                        mark_l.append(k0)

            if not prob_l:
                T1[state_num[k1],pos + 1] = -np.inf
                T2[state_num[k1],pos + 1] = None
            else:
                T1[state_num[k1],pos + 1] = max(prob_l)
                T2[state_num[k1],pos + 1] = state_num[mark_l[prob_l.index(max(prob_l))]]
    
    lastcol = list(T1[:,-1])
    # print(T1,T2)
    # print(lastcol)
    assert max(lastcol) != -np.inf, "No sequent path can be found in string: {}".format(string)
    lastmark = lastcol.index(max(lastcol))
    
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



def get_terms_vtb(string, single = False):
    '''从标注完的字符串中，找到组合起来的领域新词
    string: 字符串
    single: 是否将未组合的单个名词也加入新词组中
    '''
    r = viterbi(string)
    pairs = r[1]
    marks = r[0]
    if not marks:
        print("This sentence can not be decomposed: \n",string)
        return
    
    pairs= [[pairs[i].word, marks[i]] for i in range(len(pairs))]
    b = False
    m = False
    new_words = []
    word_parts = []
    for i in range(len(pairs)):
        if pairs[i][1] == "b" and not b: # b
            b = True
            word_parts.append(pairs[i][0])
        
        if pairs[i][1] == "m":
            # if i == (len(pairs) - 1): # bm,
            #     word_parts.append(pairs[i][0])
            #     new_words.append("".join(word_parts))
            # else:
            m = True
            word_parts.append(pairs[i][0]) # bm / ,me
                
        if pairs[i][1] == 'e' and (b or m): # be / ,me
            word_parts.append(pairs[i][0])
            new_words.append("".join(word_parts))
            
            b = False
            m = False
            word_parts = []
    
    if single == True: # only filter out the component terms  --- BME/BE/BM
        for i in range(len(pairs)):
            if pairs[i][1] == "sn":
                new_words.append(pairs[i][0])
    
    assert new_words, "There is no new term can be found in string: {}".format(string)
    return list(set(new_words))


if __name__ == "__main__":
    a = get_terms_vtb('通用机打发票的有效期')
    print(a)