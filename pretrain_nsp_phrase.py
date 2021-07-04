import tensorflow as tf
import numpy as np
import math
from BERT import bert,return_accuracy
from LOADDATA import get_corpus_indices,data_format,get_data,build_vocab,bild_vocab_token,nsp_load_data,build_vocab_label
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight

def train_nsp(model,params, train_vocab,train_labels,vocab_size,label_size,chars_to_idx,label_char_to_idx,batch_size,clip_norm):
    
    iter_data=get_data(train_vocab,train_labels,chars_to_idx,label_char_to_idx,batch_size,nsp=True)
    outputs=[]
    Ys=[]
    acc=[]
    for x,y in iter_data:
        
    
        segment_embedding_repeats=[]
        for i in range(len(x)):
            
            sep_idx=np.where(np.asarray(x[i])==chars_to_idx["[sep]"])
            segment_embedding_repeats.append((sep_idx[0]+1).astype(np.int32).tolist())
       
        label=[[l2 for l2 in l1]for l1 in y]
        X,Y,_=data_format(x,y)#格式化数据
        X=tf.concat(X,0)
        Y=tf.concat(Y,0)
        
        X=tf.one_hot(X,vocab_size)
        Y=tf.one_hot(Y,label_size)
        label=tf.one_hot(label,label_size)
        
        Y=tf.reshape(Y,[Y.shape[0]*Y.shape[1],Y.shape[-1]])
        with tf.GradientTape() as tape:
            tape.watch(params)
            output=model(X,segment_embedding_repeats=segment_embedding_repeats,next_sentence_prediction=True)
            
            loss=model.loss(output,label)
            print("loss:",np.array(loss).item())
        grads=tape.gradient(loss,params)
        
        model.update_params(grads,params)
        
        ac=return_accuracy(output,tf.reshape(label,[-1,2]),label.shape[0])
        
        acc.append(ac)
        
    
    mean_acc=tf.math.reduce_mean(acc).numpy()
    print("训练准确率：%f"%mean_acc)    
    filepath="acc.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(mean_acc)+"\n")
    flie.close()



if __name__ == "__main__":
    
    batch_size=2

    input_nums=24

    num_hiddens=4

    num_outputs=24

    layer_nums=12

    multi_head=12
    
    max_position_dim=512

    clip_norm=1.0

    datas,labels=nsp_load_data("data","nsp_data.csv","nsp_label.csv")

    num_datas=len(datas)
    
    train_vocab,train_labels,test_labels,test_vovab=datas[0:math.ceil(num_datas*0.7)][:],labels[0:math.ceil(num_datas*0.7)][:],datas[math.ceil(num_datas*0.7):-1][:],labels[math.ceil(num_datas*0.7):-1][:]

    label_idx_to_char,label_char_to_idx,label_vocab_size=build_vocab_label("data","nsp_label.csv")
    
    idx_to_chars,chars_to_idx,vocab_size=bild_vocab_token('data\\chinese_token.csv')
    
    model=bert(lr=1e-4,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,multi_head=multi_head,layers_encoder=layer_nums,vocab_size=vocab_size,pretrain=True)

    params=model.get_params()

    params_nsp=params+model.get_params_nsp()
    
    epochs=3000

    isContinue=True

    if isContinue==True :
        load_weight("ckp","params_nsp",params_nsp)
        load_weight("ckp","params",params)
    
    for i in range(epochs):
        train_nsp(model,params_nsp, train_vocab,train_labels,vocab_size,label_vocab_size,chars_to_idx,label_char_to_idx,batch_size,clip_norm)
        save_weight("ckp","params_nsp",params_nsp)
        save_weight("ckp","params",params)
        
