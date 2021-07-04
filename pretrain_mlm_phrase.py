import tensorflow as tf
import numpy as np
import math
from BERT import bert,return_accuracy
from LOADDATA import get_corpus_indices,data_format,get_data,build_vocab,bild_vocab_token
from LOAD_SAVE_PARAMS.LOAD_SAVE_PARAMS import save_weight,load_weight
#np.set_printoptions(threshold = 1e6)
def train_mlm(model,params, train_vocab,train_labels,vocab_size,label_size,chars_to_idx,label_idx_to_chars,batch_size,clip_norm):
    
    acc=[]
    iter_data=get_data(train_vocab,train_labels,chars_to_idx,label_idx_to_chars,batch_size,mlm=True)
    outputs=[]
    Ys=[]
    for x,y in iter_data:

        x_mask=model.mask(x,chars_to_idx['[mask]'])
        
        X_mask,Y,_=data_format(x_mask,y)#格式化数据

        X,Y,_=data_format(x,y)#格式化数据
        
        X=tf.concat(X,0)

        X_mask=tf.concat(X_mask,0)

        X=tf.one_hot(X,vocab_size)

        X_mask=tf.one_hot(X_mask,vocab_size)
        
        with tf.GradientTape() as tape:
            tape.watch(params)
            
            output=model(X_mask,masked_LM=True)

            loss=model.loss(output,X)
            
        grads=tape.gradient(loss,params)
        
        grads,globalNorm=tf.clip_by_global_norm(grads, clip_norm)#梯度裁剪
       
        model.update_params(grads,params)
        print("loss:",np.array(loss).item())

        output,X=tf.reshape(output,[output.shape[0]*output.shape[1],output.shape[-1]]),tf.reshape(X,[X.shape[0]*X.shape[1],X.shape[-1]])

        ac=return_accuracy(output,X,X.shape[0])
        print("训练准确率：%f"%ac)
        acc.append(ac)
        
        
    filepath="acc.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(acc).numpy())+"\n")
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
    
    train_vocab,train_labels,test_labels,test_vovab,_,_,_,label_idx_to_chars,label_chars_to_idx,label_size=build_vocab('data//data_single.csv',"evaluation","label")
    
    idx_to_chars,chars_to_idx,vocab_size=bild_vocab_token('data\\chinese_token.csv')
   
    model=bert(lr=1e-4,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,multi_head=multi_head,layers_encoder=layer_nums,vocab_size=vocab_size,pretrain=True)

    params=model.get_params()
    
    params_mlm=params+model.get_params_mlm()

    epochs=3000

    isContinue=True

    if isContinue==True:
        load_weight("ckp","params_mlm",params_mlm)
        load_weight("ckp","params",params)
    
    for i in range(epochs):
        train_mlm(model,params_mlm, train_vocab,train_labels,vocab_size,label_size,chars_to_idx,label_chars_to_idx,batch_size,clip_norm)
        save_weight("ckp","params_mlm",params_mlm)
        save_weight("ckp","params",params)
        

        
