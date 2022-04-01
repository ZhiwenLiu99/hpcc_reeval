import os
import re
import pickle
import numpy as np
import pandas as pd

from gensim import corpora
from gensim.models import LdaModel


def sample_time(cdf, tag):
    percent = list(cdf['percentile'].values)
    bounds = cdf['cdf'].values
    idx = percent.index(tag)
    x0 = percent[idx-1]
    y0 = bounds[idx-1]
    x1 = tag
    y1 = bounds[idx]
    x = np.random.uniform(low=x0, high=x1, size=None)
    return y0 + (y1-y0)/(x1-x0)*(x-x0)

def sample_size(cdf, tag):
    percent = list(cdf['percentile'].values)
    bounds = cdf['cdf'].values
    idx = percent.index(tag)
    x0 = percent[idx-1]
    y0 = bounds[idx-1]
    x1 = tag
    y1 = bounds[idx]
    x = np.random.uniform(low=x0, high=x1, size=None)
    return y0 + (y1-y0)/(x1-x0)*(x-x0)

if __name__ == "__main__":
    np.random.seed(2022)

    time_limit = 3000.0  # 时间窗口大小
    nhost = None          # 节点数量（默认值）
    load = None           # 平均流量load（默认值）

    ###############  load model  ###############
    dictionary = corpora.Dictionary.load("./model/qzone.dict")      
    doc_topics = pd.read_csv("./model/res_document_topics.csv", index_col=[0]).values  # document-topic matrix
    doc_topics = np.divide(doc_topics, np.sum(doc_topics, axis=1).reshape(-1,1))       # normalize each row to sum to 1
    model = LdaModel.load("./model/qzone.model", mmap='r')
    topic_terms = model.get_topics()
    topic_terms = np.divide(topic_terms, np.sum(topic_terms, axis=1).reshape(-1,1))    # normalize each row to sum to 1

    ###############  load cdf  ###############
    cdf_size = pd.read_csv("./stats/cdf_size.csv", index_col=[0])
    cdf_interval = pd.read_csv("./stats/cdf_interval.csv", index_col=[0])
    
    ###############  load src-dst  ###############
    num_perPair = pd.read_csv("./stats/num_perPair.csv", index_col=[0])
    pairs = num_perPair['src_dst_id'].values
    unique_ip = list(num_perPair['src_id'].values)
    unique_ip.extend(list(num_perPair['dst_id'].values))
    unique_ip = list(np.unique(unique_ip).astype('int'))
    nhost = len(unique_ip)
    print("number of hosts: {}".format(nhost))

    new_docs = []
    this_topic = np.zeros(topic_terms.shape[1])
    for i in range(len(pairs)):
        flow_id = 0
        timestamp = 0
        pair = pairs[i]
        theta = doc_topics[i]
        # print("--------------- pair: {}, start: {}, end: {} ---------------".format(pair, timestamp, time_limit))
        while timestamp<time_limit:
            # sample topic index , i.e. select topic
            z = np.argmax(np.random.multinomial(1, theta))
            # sample word from topic
            beta = topic_terms[z]
            maxidx = np.argmax(np.random.multinomial(1, beta))
            new_word = dictionary[maxidx]
            meta = re.split(',|\(|\)', new_word)  # ['', ' 65', '25', '']

            src_id = unique_ip.index(int(pair.split('_')[0]))
            dst_id = unique_ip.index(int(pair.split('_')[1]))
            interarrival = sample_time(cdf_interval, int(meta[1]))
            timestamp += interarrival
            flow = [src_id, dst_id, timestamp, sample_size(cdf_size, int(meta[2]))]
            new_docs.append(flow)
            flow_id += 1

    df = pd.DataFrame(new_docs, columns=['src_id', 'dst_id', 'start_t', 'flow_size'])
    df.sort_values(by=['start_t'], inplace=True)
    df.to_csv("./flows.csv")


    import pdb
    pdb.set_trace()