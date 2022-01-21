import jieba.analyse
import jieba.posseg as pseg
import re
import numpy as np
import pickle as pkl
import time
import configparser
import os

# 是否使用停用词
cwd = os.path.split(os.path.realpath(__file__))[0]
conf = configparser.ConfigParser()
conf.read(os.path.join(cwd , r"config.txt"))
keep_stopwords = conf.getboolean("strategy","keep_stopwords")


def add_old_terms(filepath, freq = np.e**98, tag = "n"):
    """add the old terms already had to jieba tokenizer in your specific domain 
    filepath = the relative path the file be
    freq = the terms' frequence. better be a large number for jieba to tokenize the terms in priority
    tag = set to "n" for the convinience of recognization in the "BMES" steps
    """
    with open(filepath, 'r', encoding = 'utf-8') as w:
        Terms = []
        terms = w.readlines()
        for t in terms:
            if re.findall(r'[\u4e00-\u9fa50-9]+', t.strip()):
                Terms.append(re.findall(r'[\u4e00-\u9fa50-9]+', t.strip()))

    #加入新词 
    addw = [o for i in Terms for o in i]
    for a in addw:
        jieba.add_word(a, freq = freq, tag = tag)

    return 


def clean_corpus(filepath, colname, keep_stopwords = keep_stopwords):
    """clean the raw corpus into a standard form
    filepath: relative path
    colname: the columns' name the corpus be
    """
    with open(filepath, 'rb') as rf:
        ori_corp = pkl.load(rf)

    with open(r"./data/stopwords_chi.txt", 'r', encoding= 'utf8') as sw:
        stopwords = [i.strip() for i in sw.readlines()]

    corp_l = [",".join(re.findall(r'[\u4e00-\u9fa5]+', i)) for i in ori_corp[colname].values] # only keep the chinese/others language, separated by comma
    
    if keep_stopwords: # True
        return [[o for o in pseg.lcut(i)] for i in corp_l]
         # cut the sentences with words' attribute/flag
    else:
        return [[o for o in pseg.lcut(i) if o.word not in stopwords] for i in corp_l]

def BMES_marker(filepath, colname):
    '''mark "BMES" to words for HMM
    filepath: relative file path
    colname: the columns' name the corpus be
    '''
    corp_list = clean_corpus(filepath, colname)
    corp_ll = []
    # mark标注逻辑
    # 若本身为名词类属性，则直接mark，若名词后跟一个动词类类字符，则也设为mark，组成主谓结构
    for i in corp_list:
        sub_corp = []
        mark_label = []
        last_ori = None
        for idx in range(len(i)):

            # ————————————————————————————————————————————————————————————————————
            if idx == 0: # 句子内首个词语
                if "n" in i[idx].flag or "l" in i[idx].flag:
                    sub_corp.append([i[idx].word, "mark"])
                    mark_label.append(True)
                    last_ori = i[idx].flag 
                    continue
                else:
                    sub_corp.append([i[idx].word, "s"])
                    mark_label.append(False)
                    last_ori = i[idx].flag
                    continue
                    
            # ————————————————————————————————————————————————————————————————————
            if i[idx].flag == "x": # comma ? or other symbols
                sub_corp.append([i[idx].word, 'c'])
                mark_label.append(False)
                last_ori = i[idx].flag
                continue
            elif i[idx].flag in ['ns','nr']: # 后期可添补
                sub_corp.append([i[idx].word, "s"])
                mark_label.append(False)
                last_ori = i[idx].flag
                continue

            # ————————————————————————————————————————————————————————————————————
            if "n" in i[idx].flag or "l" in i[idx].flag: # 如果词性为名词类、暂用词类，则为mark
                sub_corp.append([i[idx].word, "mark"])
                mark_label.append(True)
                last_ori = i[idx].flag

            elif mark_label[idx-1] and "v" not in i[idx].flag: # 如果前一位为mark，且该位置不是也不是名词、动词类，则single
                sub_corp.append([i[idx].word, "s"])
                mark_label.append(False)
                last_ori = i[idx].flag

            elif mark_label[idx-1] and "v" in i[idx].flag and last_ori not in ["vg","vd","v"]: #  如果前一位为mark，且为动词类，则mark。 ！！！后期看效果是否将该v也设置为s
                sub_corp.append([i[idx].word, "mark"])
                mark_label.append(True)
                last_ori = i[idx].flag

            elif "v" in i[idx].flag: # 如果前一位不是mark，且为动词，为s
                sub_corp.append([i[idx].word, "s"])
                mark_label.append(False)
                last_ori = i[idx].flag

            else: # 其他情况都为single
                sub_corp.append([i[idx].word, "s"])
                mark_label.append(False)
                last_ori = i[idx].flag

        if sub_corp:
            corp_ll.append(sub_corp)

        # for o in i:
        #     if o.word == ",":
        #         sub_corp.append([o.word, "c"])
        #         continue
        #     elif o.flag in ["ns",'nr']: # 人名、地名为s，不考虑进专业短语部分
        #         sub_corp.append([o.word, "s"])
        #         continue
            
        #     if "n" in o.flag or "l" in o.flag: # 如果词性为名词类、暂用词类，则为mark
        #         sub_corp.append([o.word, "mark"])
        #         count_m = True
                
        #     elif count_m and "v" not in o.flag: # 如果前一位为mark，且该位置不是也不是名词、动词类，则single
        #         sub_corp.append([o.word, "s"])
        #         count_m = False
                
        #     elif count_m and "v" in o.flag: # 如果前一位为mark，且为动词类，则mark。 ！！！后期看效果是否将该v也设置为s
        #         sub_corp.append([o.word, "mark"])
        #         count_m = True
                
        #     elif "v" in o.flag: # 如果前一位不是mark，且为动词，为s
        #         sub_corp.append([o.word, 's'])
        #         count_m = False
                    
        #     else: # 其他情况均为single
        #         sub_corp.append([o.word, "s"])
        #         count_m = False

        # if sub_corp:
        #     corp_ll.append(sub_corp)

    # BME标注逻辑
    for s in corp_ll:
        mark_head = False
        for i in range(len(s)):
    #         # 如果走到了最后一位
            if i == len(s)-1 and s[i][1] == "mark" and s[i-1][1] not in ['b','m']: #走到最后一位、自身为mark，但前一位没有mark
                s[i][1] = "sn"
                continue
            if i == len(s)-1 and s[i][1] == "mark" and s[i-1][1] in ['b','m']: #走到最后一位、自身为mark，但前一位没有mark
                s[i][1] = "e"
                continue
            if i == len(s)-1 and s[i][1] != "mark": #走到最后一位，但是自身不为mark，直接进入下一个sentence循环
                continue
    
            # if i == len(s)-1 and s[i][1] == "mark" and not mark_head:
            #     s[i][1] = "sn"
            #     continue
            # if i == len(s)-1 and s[i][1] == "mark" and mark_head:
            #     s[i][1] = "e"
            #     continue
            # elif i == len(s)-1 and s[i][1] != "mark":
            #     continue

            if not mark_head and s[i+1][1] == "mark" and s[i][1] == "mark": # 自己为第一个mark
                s[i][1] = "b"
                mark_head = True
            elif mark_head and s[i+1][1] == "mark" and s[i][1] == "mark":
                s[i][1] = "m"
            elif mark_head and s[i+1][1] != "mark" and s[i][1] == "mark":
                s[i][1] = "e"
                mark_head = False
            elif not mark_head and s[i][1] == "mark":
                s[i][1] = "sn"


    return corp_ll


if __name__ == "__main__":
    cwd = os.path.split(os.path.realpath(__file__))[0]
    datapath = os.path.join(cwd,"data")
    conf = configparser.ConfigParser()
    conf.read(os.path.join(cwd , r"config.txt"))
    date = time.localtime(time.time())[0:3]
    corp_file =  "clean_corp_{}.pkl".format("-".join([str(i) for i in date]))
    conf['data_name']["corp_file"] = corp_file

    if conf.get("data_name","old"):
        add_old_terms(filepath= os.path.join(datapath,conf.get("data_name","old"))) # tax_term
    
    corp =  BMES_marker(filepath = os.path.join(datapath,conf.get("data_name","new")), colname = conf.get("data_name","corp_col"))
    pkl.dump(corp, open(os.path.join(datapath,corp_file), 'wb'))
    
    # 新语料文件名称保存，用以下个模块的自动读取
    with open("config.txt", "w") as file:
        conf.write(file)
