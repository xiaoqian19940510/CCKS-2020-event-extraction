import json
import pandas as pd

def check_num_type():
    lst = []
    in_file = open('/home/LAB/liqian/test/game/Fin/dataset/trans_train.json','r')
    for line in in_file:
        line = line.strip()
        line = json.loads(line)
        #print(line)
        ids = line['id']
        content = line['content']
        for k in line['events']:
            evn_type = k['type']
            lst.append(evn_type)
    lst = set(lst)
    print(lst)
    
def change_data():
    in_file = open('/home/LAB/liqian/test/game/Fin/dataset/trans_train.json','r')
    final_lst = []
    for line in in_file:
        #org_lst = ['收购','担保','中标','签署合同','判决']
        org_lst = ['收购','判决']
        line = line.strip()
        line = json.loads(line)
        #print(line)
        ids = line['id']
        content = line['content']
        lst = []
        for k in line['events']:
            evn_type = k['type']
            lst.append(evn_type)
        #print(ids,content,lst)
        label_lst = []
        label_lst.append(ids)
        label_lst.append(content)
        for i in org_lst:
            if i in lst:
                label_lst.append(1)
            else:
                label_lst.append(0)
        #print(label_lst)
        final_lst.append(label_lst)
    return final_lst

def get_cls_train_data():
    final_lst = change_data()
    df = pd.DataFrame()
    df = df.append(final_lst,ignore_index=True)
    #df.columns = ['id','content','zy','gfgqzr','qs','tz','ggjc']
    df.columns = ['id','content','sg','pj']
    df.to_csv('/home/LAB/liqian/test/game/Fin/CCKS-Cls/pybert/dataset/train_sample.csv',index=0)
    print('分类模型训练集已转换完成！')
    
def get_cls_test_data():
    test_df = open('/home/LAB/liqian/test/game/Fin/dataset/trans_dev.json')
    lst=[]
    for line in test_df:
        line = line.strip()
        line = json.loads(line)
        #print(line)
        lst.append(line)
    df = pd.DataFrame(lst)
    df = df[['id','content']]
    df.to_csv('/home/LAB/liqian/test/game/Fin/CCKS-Cls/pybert/dataset/test.csv',index=0)
    print('分类模型测试集已转换完成！')
    
if __name__ == '__main__':
    get_cls_train_data()
    get_cls_test_data()
