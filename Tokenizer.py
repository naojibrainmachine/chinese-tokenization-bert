import tensorflow as tf
import numpy as np
import math
from BERT import bert,return_accuracy
#(lr=1e-4,input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,max_position_dim=max_position_dim,multi_head=multi_head,layers_encoder=layer_nums,vocab_size=vocab_size,label_char_to_idx=label_char_to_idx,labels_num=len(label_char_to_idx.keys()))

class tokenizer:
    def __init__(self,lr,input_nums,hidden_nums,output_nums,max_position_dim,vocab_size,label_char_to_idx,layers_encoder=12,labels_num=2,multi_head=12):
       self.bert_layer=bert(lr,input_nums=input_nums,hidden_nums=hidden_nums,output_nums=output_nums,max_position_dim=max_position_dim,vocab_size=vocab_size,layers_encoder=layers_encoder,labels_num=labels_num,multi_head=multi_head)
       

       self.opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

       


    def __call__(self,x):
        
        self.output=self.bert_layer(x)#x:(batch_size,seq_len,vocab_size)

        return self.output
        
            

    
    def predict(self,output,mask):

        if output is None:
            return self.crf_layer.vtiterbi_decode(self.output,mask)
        
        return self.crf_layer.vtiterbi_decode(output,mask)

    def get_params_bert(self):
        params=[]

        params.extend(self.bert_layer.get_params())

        return params
    def get_patams_cls(self):

        params=[]

        params.extend(self.bert_layer.get_params_tokenize())

        return params
    
    def update_params_bert(self,grads,params):

        #self.bert_layer.update_params(grads,params)
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))
        
    

    def bert_loss(self,y_pre,y_true):

        return -1*tf.reduce_mean(tf.multiply(tf.math.log(tf.nn.softmax(y_pre)+ 1e-10),y_true))
    def mask(self,x,mask_code):
        new_x=[]
        for i in range(len(x)):
            
            x_old=np.asarray(x[i])

            x_old_clone=np.asarray(x[i])

            mask_indincs_1=np.random.binomial(1, 1-0.15, len(x[i]))#把15%数据标记为0

            x_old=mask_indincs_1*x_old

            zero_indincs=np.where(x_old==0)
            
            mask_indincs_2=np.random.binomial(1, 0.8, zero_indincs[0].shape[0])

            mask_indincs_2=mask_indincs_2*mask_code#把需要mask的数据标记上mask_code

            x_old[zero_indincs]=mask_indincs_2

            zero_indincs=np.where(x_old==0)
            
            mask_indincs_3=np.random.binomial(1, 0.5, zero_indincs[0].shape[0])

            x_old[zero_indincs[0]]=mask_indincs_3*x_old_clone[zero_indincs[0]]#15%的数据中，维持10%的数据

            zero_indincs=np.where(x_old==0)
            
            if zero_indincs[0].shape[0]>0:#替换
                mask_indincs_4=np.random.randint(len(x[i]), size=zero_indincs[0].shape[0])
                #print(mask_indincs_4)
                x_old[zero_indincs]=x_old_clone[mask_indincs_4]

            new_x.append(x_old.tolist())
            
        return new_x
