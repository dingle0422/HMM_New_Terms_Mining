from re import L
import jieba.posseg as pseg
import pickle
import os
### 使得转译到vscode的中文可显示
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def check_odd(filepath, odd_word: str, split_parts = []):
    """
    查询奇怪新词的真实语料情况，若找到前后间有意义的行业新词，则以补丁形式将该词手动导入tax_terms.txt. 该方法作为模型未成熟阶段的人工补偿机制
    filepath: 带挖掘语料地址 list of str
    odd_word: 奇怪的新词 str
    split_parts:  若想查询该新词被结巴切分的详细词性结构，则将其拆成2个字段进行查询。若无法确定，则忽略 [str, str]
    """
    
    a = pickle.load(open(filepath, 'rb'))
    res = []
    for i in a:
        if odd_word in i:
            res.append(i)
            if split_parts:
                if split_parts[0] in [i.word for i in pseg.lcut(i)] and split_parts[1] in [i.word for i in pseg.lcut(i)]:
                    print(pseg.lcut(i), "\n")
    print(res)


def add_odd(words: list):
    """
    往tax_terms.txt加入新术语，作为人工补偿机制
    words: list of odd words
    """
    path = os.path.join(cwd ,r"data/tax_terms.txt")
    with open(path, 'a', encoding= 'utf8') as t:
        t.writelines([i + "\n" for i in words])

    return



if __name__ == "__main__":

    cwd = os.path.split(os.path.realpath(__file__))[0]
    filepath = os.path.join(cwd ,r"Mining/TTT.pkl")
    check_odd(filepath, '报主表',['报','主表']) # 账手册、税办法、钱误差、人名单、税门户、书号码、报主表
    add_odd(['企税','季报','月报','年报','主表'])