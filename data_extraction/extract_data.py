# 对话日志拉取
from pyhive import hive
from sqlalchemy import create_engine
import pickle
import json
import requests
import os
import pandas as pd
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


os.system("echo '10.64.6.6 hz-hadoop-nn-64-6-6' >>/etc/hosts")
os.system("echo '10.64.6.5 hz-hadoop-nn-64-6-5' >>/etc/hosts")
os.system("echo '10.98.5.5 hz-hadoop-nn-98-5-5' >>/etc/hosts")
os.system("cp /root/krb5.conf /etc/")
os.system("apt-get install krb5-user -y")
os.system("kinit -k -t /root/jrsyb.keytab jrsyb@HADOOP.COM")

# hive_url = 'hz-hadoop-nn-98-5-5'
# hive_user = 'jrsyb'
# connection = hive.Connection(host=hive_url, port=10000, auth="KERBEROS",kerberos_service_name="hive",username=hive_user)

# content = pd.read_sql('''select * from servyou_ods.ods_edw051_chat_content_df where pt_d = '2021-10-31' and store_time>'2020-10-1' '''
#                       ,connection, chunksize=10000)
# dat = pd.DataFrame()
# for idx,val in enumerate(content):
#     val.to_csv("/root/playground/HMM_New_Term_Mining/data/{}.csv".format(idx))#./datas_20.10.1-21.11.1/content



# # 对话日志整理
# import pandas as pd
# import re
# import os
# from tqdm import tqdm
# import numpy as np
# regex = re.compile(u'<.*?>')
# ds = []
# # def get_kefu(text):
# #     t = [i[9:-3] for i in re.findall("<I.*?>(.*?)</I>",text)]
# #     if len(t)==0:
# #         t = [i[9:-3] for i in re.findall("<IA.*?>(.*?)</IA>",text)]
# #     return t
# for idx in tqdm(range(4)):#483
# # for idx in tqdm(range(len(os.listdir("datas_20.10.1-21.11.1/content/")))):
#     df = pd.read_csv("/root/playground/HMM_New_Term_Mining/data/{}.csv".format(idx))
#     df.pop("Unnamed: 0")
#     df.columns = [i.split(".")[1] for i in df.columns]
#     df = df[["operator_name","content","close_reason","inner_id","msg_id"]]
#     df = df[df["content"].notnull()]
# #     print(df[~df["msg_id"].isin(df_zhuanjie)].shape,df[df["msg_id"].isin(df_zhuanjie)].shape)
# #     df = df[~df["msg_id"].isin(df_zhuanjie)]
#     kefus = []
#     users = []
#     for content in df["content"]:
#         content= content.replace("CDATA[已关闭智能助理]","").replace("\n","").replace("\t","")
#         ori_d = re.findall("CDATA\[(.*?)\]",content)
#         ori_c = re.findall("<he|<I|<IA|<offline",content)
#         if len(ori_c)<=1 or len(ori_d)<=1:
#             kefus.append([])
#             users.append([])
#             continue
#         new_c = [ori_c[0]]
#         new_d = [ori_d[0]]
#         flag = ori_c[0]
#         idx = 1
#         while idx<len(ori_c) and idx<len(ori_d):
#             if ori_c[idx]==flag:
#                 new_d[-1]+="。"+ori_d[idx]
#                 idx+=1
#             else:
#                 new_d.append(ori_d[idx])
#                 new_c.append(ori_c[idx])
#                 flag = ori_c[idx]
#                 idx+=1
#         new_d = np.array(new_d)
#         new_c = np.array(new_c)    
#         kefus.append(list(new_d[new_c!="<he"]))
#         users.append(list(new_d[new_c=="<he"]))
#     df["user"] = users
#     df["kefu"] = kefus

#     ds.append(df)
# #     df.to_csv("datas_20.10.1-21.11.1/processed-1/{}.csv".format(idx),index=None)
# df = pd.concat(ds)
# # 首句关键问提取+清洗
# df = df[["content","msg_id","user","kefu"]]
# filter_sentence = ["亿企咨询，有温度更有价值","有财税问题，上亿企咨询","财税问题，上亿企咨询",
#                "美好的一天从遇见您开始","很高兴为您解答财税问题","很高兴为您","美好的一天","转接成功","财税问题全方位指导"]
# tmp = []
# c = 0
# for idxx,i in enumerate(df["user"].values):
#     flag = ""
#     for j in i:
#         kefu_flag = False 
#         for x in filter_sentence:
#             if x in j:
#                 kefu_flag = True
#                 print(idxx)
#                 c+=1
#                 break
            
#         if kefu_flag:
#             continue
            
#         j = j.lower() 
#         regex = re.compile((u'\u4e00-\u9fa5'))
#         t = regex.sub('', j)
#         t = t.replace(" ","")
#         j = re.sub("<.*?>","",j)
#         s = "咨询.问题|什么原因.|是这样.吗|不好意思|我.问.下|问.问题|有.问题|请教.问题|我上面|您看.下|人工服务|转人工|转接人工|人工|客服|好的|谢谢|\
#         帮我看看|请问.下|请问|可以了吗｜一下|请教|嗯|麻烦|看看|我的问题|咨询.下|laoshi|老师|你好|在吗|？|，|。|！|nihao|您好|吧|zaima|在吗|呢|的|\xa0|\
#         有个问题|请教个问题|我上面|您看一下|人工服务|转人工|转接人工|人工|客服|好的|谢谢|帮我看看|请问一下|可以了吗｜\
#                    一下|请教|嗯|麻烦了|看看|我的问题|咨询一下|laoshi|老师|你好|在吗|？|，|。|！|nihao|您好|吧"
#         t = re.sub(s,"",j)
#         if len(t)>5 :
#             flag = j
#             break
#     tmp.append(flag)
# df["key"] = tmp


# 高频问拉取
def get_question(url, botId, cat=None):
    data = json.loads(requests.get(url + botId).content)
    question = []
    standard_id = []
    question_type = []
    for item in data['value']:
        question.append(item['question'])
        standard_id.append(item['standardId'])
        question_type.append(item['questionType'])
    return pd.DataFrame(data={"标准问题": standard_id, "question": question, "questionType": question_type})

bot_id = "581138583458156544" # 机器人高频问语料，可以在 bot-knowledge.dc.servyou-it.com 找到
prefix = "http://bot-knowledge.dc.servyou-it.com/{api}/modelData/service/"
df_high_freq = get_question(prefix,bot_id)
Q_hf = pd.DataFrame(df_high_freq['question'])
Q_ori = pickle.load(open(r"/root/playground/HMM_New_Term_Mining/data/df_question_0.pk", 'rb'))

Q_all = Q_hf.append(Q_ori)
pickle.dump(Q_all, open(r"/root/playground/HMM_New_Term_Mining/data/question_add_hf.pkl", 'wb'))