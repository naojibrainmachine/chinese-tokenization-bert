import tensorflow as tf
import numpy as np
import math
from LOADDATA import get_corpus_indices,data_format,get_data,build_vocab,ner_load_data,bild_vocab_token,build_vocab_label
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight
from Tokenizer import tokenizer

def action(model,params, data,vocab_size):
    data=[data]
    X,_,mask=data_format(data,None)

    mask=tf.concat(mask,0)
    
    X=tf.one_hot(X[0],vocab_size)
    
    
    Y_pre=model(x=X)
    
    
    return Y_pre

def show_result(data,y_pre_str):

    flag=False

    targets=[]
    target=''
    flag=0
    for i in range(len(y_pre_str)):
        if y_pre_str[i]=="B" and flag==0 or y_pre_str[i]=="I" and flag==1 :
            flag=1
            target=target+data[i]
            
        elif y_pre_str[i]=="B" and flag==1:
            print(target)
            targets.append(target)
            target=""
            target=data[i]
        if i==len(y_pre_str)-1 and y_pre_str[i]=="B":
            targets.append(data[i])       
                
    return targets
    
if __name__ == "__main__":
    
    batch_size=2

    input_nums=24

    num_hiddens=4

    num_outputs=24

    layer_nums=12

    multi_head=12
    
    max_position_dim=512

    clip_norm=1.0

    idx_to_chars,chars_to_idx,vocab_size=bild_vocab_token('data\\chinese_token.csv')#用汉字字库代替build_vocab返回的数据集生成的字库

    label_idx_to_char,label_char_to_idx,label_vocab_size=build_vocab_label("data","tokenize_label_num_weiboNER_2nd_conll.train.csv")#读取label词库

    model=tokenizer(lr=1e-4,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,multi_head=multi_head,layers_encoder=layer_nums,vocab_size=vocab_size,labels_num=len(label_char_to_idx.keys()),label_char_to_idx=label_char_to_idx)

    params=model.get_params_bert()#bert的基础参数，为了恢复预训练的参数数据

    
    params_bert=params+model.get_patams_cls()
    
    epochs=3000

    isContinue=True

    if isContinue==True:
        
        load_weight("ckp","params_tokenize",params_bert)
        
    while True:
        
        sentence1=input(("请输入需要命名实体识别的文字\n"))
        sentence1=list(sentence1)
        sentence=[chars_to_idx[char] for char in sentence1]
        
        y_pre=action(model,params_bert,sentence,vocab_size)
        
        y_pre=tf.math.argmax(tf.nn.softmax(y_pre,-1),-1).numpy().tolist()
        y_pre=[label_idx_to_char[idx] for idx in y_pre[0]]
        
        targets=show_result(sentence1,y_pre)

        print(targets)
