from HMM_Model import get_terms_vtb
import pickle
import pandas as pd
import logging

logger = logging.getLogger() # 创建日志对象

logger.root.setLevel(logging.INFO) # 设置可输出日志级别范围

# console_handler = logging.StreamHandler() # 将信息输出到控制台
file_handler = logging.FileHandler(filename= "./logs/log.log", encoding = 'utf8') # 将信息保存到log文件

# 设置格式并赋予handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将handler添加到日志器中
# logger.addHandler(console_handler)
logger.addHandler(file_handler)


if __name__ == "__main__":
    # with open(r"./Mining/test_corp.txt", 'r', encoding= 'utf8')  as t:
    t = pickle.load(open(r"./Mining/TTT.pkl",'rb'))
    res = []
    for s in t: #.readlines():
        try:
            res.append(get_terms_vtb(s))
        except Exception as E:
            logger.info(E)

    pickle.dump(res, open(r"./Mining/NT.pkl", 'wb'))
    a = pickle.load(open(r"./Mining/NT.pkl", 'rb'))
    print(a,len(a))