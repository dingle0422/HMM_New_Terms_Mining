from HMM_Model import get_terms_vtb, tokenizer
from word_filter import overlap_out
import pickle
import os
import logging
import configparser


logger = logging.getLogger() # 创建日志对象
logger.root.setLevel(logging.INFO) # 设置可输出日志级别范围
# console_handler = logging.StreamHandler() # 将信息输出到控制台
file_handler = logging.FileHandler(filename= "./logs/log.log", encoding = 'utf8') # 将信息保存到log文
# 设置格式并赋予handler
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# 将handler添加到日志器中
# logger.addHandler(console_handler)
logger.addHandler(file_handler)



cwd = os.path.split(os.path.realpath(__file__))[0]
conf = configparser.ConfigParser()
conf.read(os.path.join(cwd , r"config.txt"))
keep_stopwords = conf.getboolean("strategy","keep_stopwords")


def new_att(sentence, word):
    """
    将不同的词性转化为新词的n/nv词性
    sentence: 该新词所在原语料
    word: 新词
    """
    w_att = tokenizer(sentence, keep_stopwords = keep_stopwords)
    a = set()
    for i in w_att:
        if i.word in word:
            a.add(i.flag)

    att = set()
    for i in a:
        if "l" in i:
            att.add("n")
        elif "n" in i:
            att.add("n")
        elif "v" in i:
            att.add("v")

    if "n" in att and "v" in att:
        return "nv"

    if  len(att) == 1 and "n" in att:
        return "n"



def get_terms(filepath, n_only = True, overlaps = False,one_sentence = False):
    """
    返回最终发现的新词
    filepath: 需要进行新词挖掘的文件路径.pkl
    one_sentence: 被调用时，为单个句子或txt文件
    """
    t = pickle.load(open(filepath,'rb'))
    res = []
    for s in t: 
        try:
            new_word = get_terms_vtb(s, one_sentence)
            for nw in new_word:
                NW = [nw, new_att(s, nw)]
                if NW not in res: # 防止一句话里多个相同词的提取
                    res.append(NW)

        except Exception as E:
            logger.info(E)

    if n_only == True:
        e = []
        for i in res:
                if i[1] == "n":
                    e.append(i)

        if overlaps:
            return e
        else:
            return overlap_out(e)

    else:
        if overlaps:
            return res
        else:
            return overlap_out(res)






if __name__ == "__main__":
    # # with open(r"./Mining/test_corp.txt", 'r', encoding= 'utf8')  as t:
    cwd = os.path.split(os.path.realpath(__file__))[0]
    filepath = os.path.join(cwd , r"Mining/TTT.pkl")
    
    res = get_terms(filepath, n_only = True, overlaps= False)
   
    
    print(res,len(res))
    pickle.dump(res, open(os.path.join(cwd ,r"Mining/NT.pkl"), 'wb'))

    # a = pickle.load(open(os.path.join(cwd ,r"Mining/NT.pkl"), 'rb'))
    # print(get_terms_vtb("选择", one_sentence=False, single= True))