# -*- coding:utf-8 -*-
import json
import pandas as pd
import os
from tqdm import tqdm
from data.DSTC2.scripts.dataset_walker import dataset_walker
from data.DSTC2.scripts.misc import S
from sklearn.feature_extraction.text import CountVectorizer

#####################################
# 各种文件路径
# 加载train、dev、test的数据列表
scripts_path = os.path.join('data', 'DSTC2', 'scripts', 'config')
train_list_path = os.path.join(scripts_path, 'dstc2_train.flist')
dev_list_path = os.path.join(scripts_path, 'dstc2_dev.flist')
test_list_path = os.path.join(scripts_path, 'dstc2_test.flist')

# ontology path
ontotogy_path = os.path.join(scripts_path, 'ontology_dstc2.json')

# 数据root 路径
data_root = os.path.join('data', 'DSTC2', 'data')


# feature
#####################################
def data_process(data):
    """
    :param data: string
        train: 'dstc2_train'
        dev : 'dstc2_dev'
        test: 'dstc2_test'
    :return:
    """
    dataset = dataset_walker(data, labels=True)
    turn_list = list()
    for call in tqdm(dataset):  # 一个call，一次通话，一完整的对话turning
        for log, label, log_path, label_path in call:
            turn = dict()
            # 获得数据的唯一path
            # log_list = log_path.split('\\')[-5:]
            # label_list = label_path.split('\\')[-5:]
            # turn['log_path'] = '\\'.join(log_list)
            # turn['label_path'] = '\\'.join(label_list)
            # print(turn)
            turn['log_path'] = log_path
            turn['label_path'] = label_path

            # 获得对话内容
            turn['sys_sentence'] = log['output']['transcript']
            turn['usr_sentence'] = log['input']['live']['asr-hyps'][0]['asr-hyp']  # 用户句子
            # print(turn['sys_sentence'])
            # 获得用户输入的slot + slot value
            slu = S(log)  # 返回defaultdict(set), 里面{slot_tpye
            turn['turn_slu'] = slu
            turn_list.append(turn)

    result = pd.DataFrame(turn_list)
    save_path = os.path.join('data', data + '_features.csv')
    print(result.usr_sentence.head(10))
    result.to_csv(save_path, index=False)


################################
def built_vocabulary():
    """
    :return:
    """
    dataset = ['dstc2_dev_features', 'dstc2_dev_features', 'dstc2_dev_features']
    all_turn = list()
    for data in dataset:
        path = os.path.join('data', data + '.csv')
        data = pd.read_csv(path)
        data = data.fillna('-')
        sys = data['sys_sentence'].tolist()
        usr = data['usr_sentence'].tolist()
        all_turn.extend(sys)
        all_turn.extend(usr)
        break
    # 计算vector
    vectorizer = CountVectorizer()

    vectorizer.fit_transform(all_turn)
    vocabulary = vectorizer.get_feature_names()
    print(len(vocabulary))
    print(vocabulary)





if __name__ == '__main__':
    data_set = {'train': 'dstc2_train',
                'dev': 'dstc2_dev',
                'test': 'dstc2_test'}
    # data_process(data_set['train'])
    # data_process(data_set['dev'])
    # data_process(data_set['test'])
    built_vocabulary()
