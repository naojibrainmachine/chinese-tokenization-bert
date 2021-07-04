import os
try:
    import fool
except:
    print("缺少fool工具")
import math
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import re

np.random.seed(1)

def add2vocab(path,word):
    vocab_data=pd.read_csv(path)
        
    idx_to_chars=list(vocab_data['vocabulary'])+[word]

    df_data = pd.DataFrame(idx_to_chars, columns=['vocabulary'])

    df_data.to_csv(path,index=0)

def get_corpus_indices(data,chars_to_idx,mlm=False,nsp=False):
    """
    转化成词库索引
    
    """
    
    corpus_indices=[]
    keys=chars_to_idx.keys()
    #print(data)
    for d in data:
       
        if nsp==True:
            
            corpus_chars=d
            corpus_chars_idx=[]
            
            if len(d)>0 and len(d[0])==1:
                corpus_chars=['[cls]']+corpus_chars

                index=-1
                for word in corpus_chars:
                    index=index+1
                    if word not in keys:
                        
                        corpus_chars[index]='[mask]'#用[mask]替换不存在词库中的单词
                        
                corpus_chars_idx=[chars_to_idx[char] for char in corpus_chars]
                
                find_end=np.where(np.asarray(corpus_chars_idx)==chars_to_idx['。'])
                for i in range(find_end[0].shape[0]):
                    
                    corpus_chars_idx.insert(find_end[0][i]+i+1,chars_to_idx['[sep]'])
            else:
                corpus_chars_idx=[chars_to_idx[char] for char in corpus_chars]
            
        elif mlm==True:
            d=d.replace('\n','').replace('\r','').replace(' ','').replace('\u3000','')
            corpus_chars=list(d)
            corpus_chars_idx=[]
            #print(2)
            '''
            index=-1
            for word in corpus_chars:
                index=index+1
                if word not in keys:
                    corpus_chars[index]='[mask]'#用[mask]替换不存在词库中的单词
            '''
            
            index=-1
            for word in corpus_chars:
                index=index+1
                if word not in keys:
                    corpus_chars[index]='[mask]'#用[mask]替换不存在词库中的单词
            corpus_chars_idx=[chars_to_idx[char] for char in corpus_chars]
        else:
            
            corpus_chars=d
            if isinstance(corpus_chars,(list)):#corpus_chars必须是列表list
                index=-1
                for word in corpus_chars:
                    index=index+1
                    if word not in keys:
                        corpus_chars[index]='[mask]'#用[mask]替换不存在词库中的单词
            else:
                corpus_chars=[corpus_chars]#转化成list
            corpus_chars_idx=[chars_to_idx[char] for char in corpus_chars]
            
        corpus_indices.append(corpus_chars_idx)#语料索引，既读入的文本，并通过chars_to_idx转化成索引
    
    return corpus_indices

def data_format(data,labels):
    '''
    数据格式化，把整个批次的数据转化成最大数据长度的数据相同的数据长度（以-1进行填充）
    '''
    def format_inner(inputs,max_size):
        
        new_data=[]
        for x_t in inputs:
            if(abs(len(x_t)-max_size)!=0):
                for i in range(abs(len(x_t)-max_size)):
                    x_t.extend([-1])
            new_data.append(tf.reshape(x_t,[1,-1]))
        return new_data
    max_size=0
    new_data=[]
    mask=[]
    masks=[]
    new_labels = []
    #获取最大数据长度
    for x in data:
        if(max_size<len(x)):
            max_size=len(x)

    #得到masks
    for d in data:
        for i in range(max_size):
            if(i<len(d)):
                mask.append(1.0)
            else:
                mask.append(0.0)
        masks.append(tf.reshape(mask,[1,-1]))
        mask=[]
    
    #print(masks,"max_size")
    if data is not None:
        new_data=format_inner(data,max_size)#格式化数据
        
    if labels is not None:
        new_labels=format_inner(labels,max_size) #格式化标签

    

    #print(new_labels)
    #print(new_data)
    return new_data,new_labels,masks



def get_data(data,labels,chars_to_idx,label_chars_to_idx,batch_size,char2idx=True,mlm=False,nsp=False):
    '''
    function:
        一个批次一个批次的yield数据
    parameter:
        data:需要批次化的一组数据
        labels:data对应的情感类型
        chars_to_idx;词汇到索引的映射
        label_chars_to_idx;标签到索引的映射
        batch_size;批次大小
    '''
    num_example=math.ceil(len(data)/batch_size)
    
    example_indices=list(range(num_example))
    random.shuffle(example_indices)
    #print(data,"get_data")
    for i in example_indices:
        start=i*batch_size
        if start >(len(data)-1):
            start=(len(data)-1)
            
        
        end=i*batch_size+batch_size
        if end >(len(data)-1):
            end=(len(data)-1)+1
        
        X=data[start:end]
        Y=labels[start:end]
        #print(chars_to_idx,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        #print(char2idx," ",mlm," ",nsp,"进1")
        if char2idx==True:
            #print("进")
            X=get_corpus_indices(X,chars_to_idx,mlm=mlm,nsp=nsp)
            if mlm==True:
                Y=X
            else:
                Y=get_corpus_indices(Y,label_chars_to_idx,mlm=mlm,nsp=nsp)
            #print(X,"XXXXXX")
        yield X,Y #只是索引化的文本，且长度不一

def nsp_vocab(folder,name):
    path=folder+"\\"+name

    df = pd.read_csv(path)

    data = list(df["evaluation"])
    #print(len(data))
    datas=[]

    labels=[]
    
    for i in range(len(data)):
        if data[i].find("。")==-1:
            continue

        #print(data[i])
        x,y=build_sample_nsp(data[i])
        
        if x==-1:
            continue
        datas.extend(x)
        labels.extend(y)
    #print(datas[-1])
    #print(labels[-1])

    datas=[list(d) for d in datas]

    df_data = pd.DataFrame(datas)
    df_data.to_csv(folder+"\\"+"nsp_data.csv",index=0,header=0)#不存列名和行名

    df_label = pd.DataFrame(labels)
    df_label.to_csv(folder+"\\"+"nsp_label.csv",index=0,header=0)#不存列名和行名
    #print(datas[-1][:])

def nsp_load_data(folder,data_name,label_name):

    path_data=folder+"\\"+data_name

    path_name=folder+"\\"+label_name

    df_data = pd.read_csv(path_data,header=None,low_memory=False)

    df_label = pd.read_csv(path_name,header=None,low_memory=False)
    
    data=df_data.values.tolist()
    #print(data[0:10][:])
    data=[[d for d in sentence if pd.notna(d)] for sentence in data ]#去除nan
    #print(df_label,"####")
    label=df_label.values.tolist()
   
    return data,label
    
def build_sample_nsp(data):
    '''
    function:
        产生由两句话组合的数据，label表示这两句话是否是在原文本连在一起的。
        比如一个文档有3句话，为：(文1)(文2)(文3)，连在一起的（正样本）是(文1)(文2)，(文2)(文3)，负样本是(文1)(文3)，(文3)(文1)，(文3)(文2)，(文2)(文1)。
        为了保证正负样本数量一致，所以该函数返回所有的样本中，正负样本各占50%
    parameter:
        data:一个文档。一个文档包含很多句子
    '''
    def dw_nsp(sentence):
        front=0
    
        back=1

        pos_data=[]

        pos_label=[]

        neg_data=[]

        neg_label=[]
        
        if(len(sentence)>1):
            while back<len(sentence):
                #print(back)
                pos_data.append(sentence[front]+"。"+sentence[back]+"。")
                pos_label.append("正面")#1代表正样本

                back=back+1
                front=front+1
                
            #print(len(pos_data),"pos_data")
            for i in range(len(pos_data)):
                front=random.randint(0,len(sentence)-1)
                back=random.randint(0,len(sentence)-1)
                #print(back)
                while (back-front)==1:
                    front=random.randint(0,len(sentence)-1)
                    back=random.randint(0,len(sentence)-1)
                #print(back)    
                neg_data.append(sentence[front]+"。"+sentence[back]+"。")
                neg_label.append("负面")#0代表负样本
        
        return pos_data+neg_data,pos_label+neg_label
    
    pattern=r'[。|！]'#以句号分割所有文本
    #print(data)
    corpus=data
    
    sentence=re.split(pattern,corpus)
    
    if len(sentence[-1])==0:
        sentence=sentence[0:-1]#由于最后一个元素是空的所以把它去除
        #print(sentence)
    
    #print(sentence)
    if len(sentence)>2:
        inputs,labels=dw_nsp(sentence)
        #print(inputs)
        #print(labels)
        #random.shuffle(zip(inputs,labels))#分开打乱会有问题
        inputs_labels = list(zip(inputs, labels))
        random.shuffle(inputs_labels)
        inputs[:], labels[:] = zip(*inputs_labels)
        
        return inputs,labels
    else:
        return -1,-1
       
    
    
def chinese_token_vocab(path):
    df = pd.read_csv(path, header=None)
    chinese_token=list(df.iloc[:,-1])+["。","？","！","，","、","；","：","’","‘","”","“","（","）","【","】","—","…","·","〔","〕","[","]",".","0","1","2","3","4","5","6","7","8","9"]+['[cls]','[sep]','[mask]']#加入一些常见中文符号

    df_data=pd.DataFrame(chinese_token, columns=['token'])
    df_data=pd.DataFrame(df_data['token'].unique(), columns=['token'])
    df_data.to_csv("data\\chinese_token.csv",index=0)

def bild_vocab_token(path):
    '''
    读取token数据集
    '''
    df = pd.read_csv(path)
    
    chinese_token=list(df['token'])

    char_to_idx=dict([(char,i) for i,char in enumerate(chinese_token)])
    
    vocab_size=len(chinese_token)

    return chinese_token,char_to_idx,vocab_size

def ner_vocab(folder,name):#tokenize
    '''
    func:
        把数据集构建成一行代表一个句子。data和label分开成两个文档。
    '''
    name_1="ner_label_num_"+name+".csv"
    name_2="ner_data_"+name+".csv"
    name_3="ner_label_"+name+".csv"
    name_4="tokenize_label_num_"+name+".csv"
    name_5="tokenize_label_"+name+".csv"
    path = folder+"\\"
    
    df = pd.read_csv(folder+"\\"+name,sep='\t', header=None,encoding='utf-8')

    corpus=list(df[0])
    corpus_label=list(df[1])
    token=[]
    tokenize=[]
    tokenizes=[]
    tokens=[]

    label=[]
    labels=[]
    #构建数据集，把token和对应的label存到一个文件里
    for i in range(len(corpus)):
        chars=list(corpus[i])
        token.append(chars[0])
        #print(chars[0],i)
        if(int(chars[1])>0):
            tokenize.append("I")
        else:
            tokenize.append("B")
            
        label.append(corpus_label[i])
        #tokenize.append(i[1])
        if chars[0]=="。":#以句号代表一句话结束
            tokens.append(token)
            labels.append(label)
            tokenizes.append(tokenize)
            #token.clear()
            #label.clear()
            token=[]
            label=[]
            tokenize=[]
    
    
    
    #data={"token":token,"tokenize":tokenize,"ner_label":ner_label}
    #print(tokens)
    df_data = pd.DataFrame(tokens)
    df_data.to_csv(path+name_2,index=0,header=0)#不存列名和行名

    df_label = pd.DataFrame(labels)
    df_label.to_csv(path+name_3,index=0,header=0)#不存列名和行名

    #print(tokenizes)
    df_label_tokenize = pd.DataFrame(tokenizes)
    df_label_tokenize.to_csv(path+name_5,index=0,header=0)#不存列名和行名
    
    #构建char到idx的映射，并保存备用
    ner_label_num=list(df[1].unique())#+["<START>","<END>"]
    df_label = pd.DataFrame(ner_label_num, columns=['label'])
    df_label.to_csv(path+name_1,index=0)

    #把嵌套的list整合成一个list
    T=[]
    for t in tokenizes:
        T.extend(t)
    
    df_label_tnum = pd.DataFrame(list(set(T)), columns=['label'])#+["<START>","<END>"]
    df_label_tnum.to_csv(path+name_4,index=0)
    
def build_vocab_label(folder,name):

    df = pd.read_csv(folder+"\\"+name)
    #print(df)

    try:
        label_idx_to_char=list(df["label"].unique())
    except:
        label_idx_to_char=list(df.iloc[:,0].unique())
    

    label_char_to_idx=dict([(char,i) for i,char in enumerate(label_idx_to_char)])

    label_vocab_size=len(label_idx_to_char)

    return label_idx_to_char,label_char_to_idx,label_vocab_size

def ner_load_data(folder,name,low_memory=False):

    path_1=folder+"\\"+name

    df = pd.read_csv(path_1,header=None,low_memory=False)
   
    data=df.values.tolist()
    
    data=[[d for d in sentence if pd.notna(d)] for sentence in data ]#去除nan
   
    return data#,labels


    
def build_vocab(path,data_name,label_name):
    """
    构建词库
    path：数据集路径
    """
    df = pd.read_csv(path)

    #打乱索引
    rand=np.random.permutation(len(df))
    
    #获取数据总条数
    num_sum=len(df[label_name])

    #获取所有数据，为构建词库做准备
    vocab = list(df[data_name])

    #获取所有标签
    labels=list(df[label_name].unique())

    #获取训练数据，所有数据的90%为训练数据
    train_labels, train_vocab = list(df[label_name].iloc[rand])[0:int(num_sum*0.9)], list(df[data_name].iloc[rand])[0:int(num_sum*0.9)]

    #获取测试数据，所有数据的10%为测试数据
    test_labels,test_vovab=list(df[label_name].iloc[rand])[int(num_sum*0.9):num_sum], list(df[data_name].iloc[rand])[int(num_sum*0.9):num_sum]
    
    idx_to_chars=[]#索引到词汇的映射
    chars_to_idx={}#词汇到索引的映射

    label_idx_to_chars=[]
    
    #构建词库，用foolnltk进行分词
    if os.path.exists("data\\idx_to_chars.csv")==False:

        for i in range(len(vocab)):
            corpus=vocab[i].replace('\n','').replace('\r','').replace(' ','').replace('\u3000','')
            #corpus_chars=fool.cut(corpus)
            #corpus_chars=corpus_chars[0]
            corpus_chars=list(corpus)
            idx_to_chars.extend(corpus_chars)
        
            
        idx_to_chars=list(set(idx_to_chars))+['[cls]','[sep]','[mask]']#索引到词汇的映射
        label_idx_to_chars=list(set(labels))#索引到标签的映射

        df_data = pd.DataFrame(idx_to_chars, columns=['vocabulary'])
        df_data.to_csv("data\\idx_to_chars.csv",index=0)

        df_label = pd.DataFrame(label_idx_to_chars, columns=['label'])
        df_label.to_csv("data\\label_idx_to_chars.csv",index=0)

        
    else:
        
        
        vocab_data=pd.read_csv("data\\idx_to_chars.csv")
        
        idx_to_chars=list(vocab_data['vocabulary'])

        vacab_label= pd.read_csv("data\\label_idx_to_chars.csv")
        label_idx_to_chars=list(vacab_label['label'])


    
    chars_to_idx=dict([(char,i) for i,char in enumerate(idx_to_chars)])#词汇到索引的映射

    label_chars_to_idx=dict([(char,i) for i,char in enumerate(label_idx_to_chars)])#标签到索引的映射

    vocab_size=len(idx_to_chars)#词库大小

    label_size=len(label_idx_to_chars)
    
    vocab.clear()

    return train_vocab,train_labels,test_labels,test_vovab,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size

#build_vocab('data//data_single.csv')
#vocabulary,labels ,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size=build_vocab('data//data_single.csv')
#get_data(data=vocabulary,labels=labels,chars_to_idx=chars_to_idx,label_chars_to_idx=label_chars_to_idx,batch_size=3)


#chinese_token_vocab("data\\chinese_token.txt")#对原始数据进行清洗
#chinese_token,char_to_idx,vocab_size=bild_vocab_token("data\\chinese_token.csv")#读取词库

#print(chinese_token,vocab_size,vocab_size)

#print(data[0:10])

#print(label)
#label_idx_to_char,label_char_to_idx,label_vocab_size=build_vocab_label("data","ner_label_num_weiboNER_2nd_conll.train.csv")#读取label词库
#print(label_idx_to_char,"label_idx_to_char")
#print(label_char_to_idx,"label_char_to_idx")
#print(label_vocab_size,"label_vocab_size")

#ner_vocab("data","weiboNER_2nd_conll.train")#构建训练集
#ner_vocab("data","weiboNER_2nd_conll.test")#构建测试集
#data=ner_load_data("data","ner_data_weiboNER_2nd_conll.train.csv")#读取训练数据
#label=ner_load_data("data","ner_label_weiboNER_2nd_conll.train.csv")#读取训练数据的标签
#print(label)


#nsp_vocab("data","data_single.csv")#构建next sentence predict数据集
#nsp_load_data("data","nsp_data.csv","nsp_label.csv")#加载next sentence predict数据集
