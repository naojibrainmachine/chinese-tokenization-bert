import tensorflow as tf
import numpy as np
from ENCODER.ENCODER import encoder
from EMBEDDING.EMBEDDING import embedding

class bert:
    def __init__(self,lr,input_nums,hidden_nums,output_nums,max_position_dim,vocab_size,layers_encoder=12,labels_num=2,multi_head=12,pretrain=False,CRF=False):#input_num=768 

        self.input_nums=input_nums
        self.max_position_dim=max_position_dim#512
        
        self.encoders=[encoder(input_nums=input_nums,hidden_nums=hidden_nums,output_nums=output_nums,multi_head=multi_head) for _ in range(layers_encoder)]
        self.embed_token=embedding(vocab_size,input_nums)
        
        self.embed_position=embedding(max_position_dim,input_nums)

        if pretrain==True:
            self.w1=tf.Variable(tf.random.truncated_normal([output_nums,vocab_size],stddev=tf.math.sqrt(2.0/(input_nums+vocab_size))))
            self.b1=tf.Variable(tf.zeros(vocab_size))

            self.w2=tf.Variable(tf.random.truncated_normal([output_nums,labels_num],stddev=tf.math.sqrt(2.0/(input_nums+labels_num))))
            self.b2=tf.Variable(tf.zeros(labels_num))
        elif CRF==True:
            self.w=tf.Variable(tf.random.truncated_normal([output_nums,labels_num],stddev=tf.math.sqrt(2.0/(input_nums+labels_num))))
            self.b=tf.Variable(tf.zeros(labels_num))
        
        self.w=tf.Variable(tf.random.truncated_normal([output_nums,labels_num],stddev=tf.math.sqrt(2.0/(input_nums+labels_num))))
        self.b=tf.Variable(tf.zeros(labels_num))
        
        self.opt=tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
        
    def __call__(self,x,segment_embedding_repeats=None,masked_LM=False,next_sentence_prediction=False):
        '''
        function:bert的主体操作
        parameter:
            x:输入数据
            y:输出数据
            segment_embedding_repeats:由第一个句子token数量加2，和第二个句子token数量加1组成。如第一个句子token数量为5，第二个句子token数量为10，则segment_embedding_repeats=[7,11]
            masked_LM：在一句话中随机选择15%的词汇用于预测。对于需要预测的词汇，80%情况下采用一个特殊符号[MASK]替换，10%情况下采用一个任意词替换，剩余10%情况下保持原词汇不变。
            next_sentence_prediction：对是否是下一句话进行预测，第一个embedding是[cls],用[sep]的embedding分隔两句话,且每个句子做后token都紧跟个[sep]
        '''
        embedd_x=self.embed_token.embedding_lookup(x)
        #embedd_y=self.embed_token.embedding_lookup(y)
        pos=range(x.shape[1])

        pos_one_hot=tf.one_hot(pos,self.max_position_dim)
        pos_one_hot=tf.reshape(pos_one_hot,[1,pos_one_hot.shape[0],pos_one_hot.shape[-1]])
        embedd_pos=self.embed_position.embedding_lookup(pos_one_hot)

        if masked_LM==True and next_sentence_prediction==False:
            embedd_x=embedd_x+embedd_pos
            #embedd_y=embedd_y+embedd_pos
            
            output_x=embedd_x
            #output_y=embedd_y
            for layer in self.encoders:
                output_x,_,_=layer(output_x)

            output=tf.matmul(output_x,self.w1)+self.b1

            return tf.nn.softmax(output)
        elif masked_LM==False and next_sentence_prediction==True:
            embedd_seg=self.segment_embedding(len(segment_embedding_repeats[0]),self.input_nums)
            #print(embedd_seg,"embedd_seg kais")
            seg=[]
            for i in range(embedd_x.shape[0]):
                #print(tf.repeat(arr,repeats=re[i],axis=0))
                seg.append(tf.repeat(embedd_seg,repeats=[segment_embedding_repeats[i][0],(embedd_x.shape[1]-segment_embedding_repeats[i][0])],axis=0))
            #embedd_seg=tf.repeat(embedd_seg,repeats=segment_embedding_repeats,axis=0)
            embedd_seg=tf.stack(seg,axis=0)
            #print(embedd_x,"xxx")
            #print(embedd_seg,"embedd_seg jieshu")
            embedd_x=embedd_x+embedd_pos+embedd_seg
            #embedd_y=embedd_y+embedd_pos+embedd_seg

            output_x=embedd_x
            #output_y=embedd_y
            for layer in self.encoders:
                output_x,_,_=layer(output_x)

            return tf.nn.softmax(tf.matmul(output_x[:,0,:],self.w2)+self.b2,axis=1)#[cls]对应的输出

        elif masked_LM==False and next_sentence_prediction==False:
            embedd_x=embedd_x+embedd_pos
            #embedd_y=embedd_y+embedd_pos
            
            output_x=embedd_x
            #output_y=embedd_y
            for layer in self.encoders:
                output_x,_,_=layer(output_x)

            return tf.matmul(output_x,self.w)+self.b#由于crf所有的计算都默认在log下取得的值，所以这里不应该归一化
        
            
        
    def segment_embedding(self,segment_num,input_nums):
        output=[]
        for i in range(segment_num):
            output.append(tf.ones([1,input_nums])*i)
        return tf.concat(output,0)
    def loss(self,output,y):
        return -1*tf.reduce_mean(tf.multiply(tf.math.log(output+ 1e-10),y))
    
    def get_params(self):
        params=[]

        params.extend(self.embed_token.get_params())

        params.extend(self.embed_position.get_params())
        
        params.extend([inner_cell for cell in self.encoders for inner_cell in cell.get_params()])

        return params
    def get_params_tokenize(self):
        
        params=[]

        params.append(self.w)

        params.append(self.b)

        return params
        
    def get_params_mlm(self):
        params=[]

        params.append(self.w1)

        params.append(self.b1)

        return params

    def get_params_nsp(self):
        params=[]
        
        params.append(self.w2)

        params.append(self.b2)

        return params
    
    def update_params(self,grads,params):
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))

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
        
def return_accuracy(temp_predict,temp_batch_label,batch_size):
    '''
    计算准确率
    '''
    
    temp_data= tf.unstack(temp_predict, axis=0)
    temp_label= tf.unstack(temp_batch_label, axis=0)
    
    temp_label2=[label for label in temp_label if tf.reduce_sum(label) -1.0==0.0]
    index=[]
    for i in range(temp_batch_label.shape[0]):
        if (tf.reduce_sum(temp_batch_label[i][:]).numpy()==0.0):
            del temp_label[i-len(index)]
            del temp_data[i-len(index)]
            index.append(i)
            
    
    temp_label=tf.stack(temp_label,0)
    temp_data=tf.stack(temp_data,0)
    
    nums=temp_data.shape[0]
    
    rowMaxSoft=np.argmax(temp_data, axis=1)+1
    rowMax=np.argmax(temp_label, axis=1)+1
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    '''
    rowMaxSoft=np.argmax(tf.reshape(temp_predict,[-1,temp_predict.shape[-1]]), axis=1)+1
    rowMax=np.argmax(tf.reshape(temp_batch_label,[-1,temp_batch_label.shape[-1]]), axis=1)+1
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    '''
    nonO=rowMaxSoft-rowMax
    #print(nonO)
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
    #print(float(batch_size))
    #print(res[0][0])
    accuracy=(float(nums)-res[0][0])/float(nums)
    
    return accuracy
