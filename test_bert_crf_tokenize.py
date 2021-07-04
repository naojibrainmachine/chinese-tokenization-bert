import tensorflow as tf
import numpy as np
import math
#from BERT import return_accuracy
from LOADDATA import get_corpus_indices,data_format,get_data,build_vocab,ner_load_data,bild_vocab_token,build_vocab_label
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight
from Tokenizer import tokenizer

def test(model,params_bert, train_vocab,train_labels,vocab_size,label_size,chars_to_idx,label_idx_to_chars,batch_size,clip_norm):
    acc=[]
    acc2=[]
    F1=[]
    iter_data=get_data(train_vocab,train_labels,chars_to_idx,label_idx_to_chars,batch_size)
    outputs=[]
    Ys=[]
    los=[]
    for x,y in iter_data:

        
        label=[[y_2 for y_2 in y_1]for y_1 in y]
        
        X,Y,mask=data_format(x,y)#还要输出个mask
        #目前还需要修改数据的产生和mask的生成
        mask=tf.concat(mask,0)
        
        X=tf.concat(X,0)
        
        X=tf.one_hot(X,vocab_size)
        
        
        Y=tf.concat(Y,0)

        
        Y_bert=tf.one_hot(Y,label_size)
        
        Y=tf.cast(Y,dtype=tf.float32)
       
        output_bert=model(X)
        
        cc=return_accuracy_2(Y_bert,output_bert,mask)
        acc2.append(cc)
        print("bert训练准确率：%f"%cc)

    filepath="test_acc_bert.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(acc2).numpy())+"\n")
    flie.close()

    
    


def return_accuracy_2(Y,Y_pre,mask):
    
    rowMaxSoft=np.argmax(tf.nn.softmax(Y_pre), axis=-1)+1
    rowMax=np.argmax(Y, axis=-1)+1
    rowMaxSoft*mask.numpy()
    rowMax=rowMax*mask.numpy()
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    
    nonO=rowMaxSoft-rowMax
    nonO=nonO*tf.reshape(mask,[1,-1]).numpy()
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
    accuracy=(float(tf.reduce_sum(mask).numpy())-res[0][0])/float(tf.reduce_sum(mask).numpy())
    
    return accuracy
    

    
if __name__ == "__main__":
    
    batch_size=2

    input_nums=24

    num_hiddens=4

    num_outputs=24

    layer_nums=12

    multi_head=12
    
    max_position_dim=512

    clip_norm=1.0

    train_vocab,train_labels=ner_load_data("data","ner_data_weiboNER_2nd_conll.train.csv"),ner_load_data("data","tokenize_label_weiboNER_2nd_conll.train.csv")#读取训练数据

    test_vovab,test_labels=ner_load_data("data","ner_data_weiboNER_2nd_conll.test.csv"),ner_load_data("data","tokenize_label_weiboNER_2nd_conll.test.csv")#读取测试数据

    idx_to_chars,chars_to_idx,vocab_size=bild_vocab_token('data\\chinese_token.csv')#用汉字字库代替build_vocab返回的数据集生成的字库

    label_idx_to_char,label_char_to_idx,label_vocab_size=build_vocab_label("data","tokenize_label_num_weiboNER_2nd_conll.train.csv")#读取label词库

    model=tokenizer(lr=1e-5,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,multi_head=multi_head,layers_encoder=layer_nums,vocab_size=vocab_size,labels_num=len(label_char_to_idx.keys()),label_char_to_idx=label_char_to_idx)

    params=model.get_params_bert()#bert的基础参数，为了恢复预训练的参数数据

    params_bert=params+model.get_patams_cls()
    
    epochs=3000

    isContinue=True
    
    if isContinue==True:
        load_weight("ckp","params_tokenize",params_bert)
    
    for i in range(epochs):
        test(model,params_bert, test_vovab,test_labels,vocab_size,label_vocab_size,chars_to_idx,label_char_to_idx,batch_size,clip_norm)
        
