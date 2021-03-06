import tensorflow as tf
import numpy as np
import math
class self_attention:
    def __init__(self,input_nums,hidden_nums,output_nums,mask=False,multi_head=1):
        '''
        function:
            初始化attention
        parameter:
            input_nums:一般是词向量大小
        '''
        self.multi_head=multi_head
        self.input_nums=input_nums
        self.mask=mask
        
        self.step=0
        self.t=0
        
        self.variables={}
        for i in range(multi_head):
            q_name="q"+str(i)
            k_name="k"+str(i)
            v_name="v"+str(i)
            
            self.variables[q_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))

            self.variables[k_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))

            self.variables[v_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))
            
        self.w0=tf.Variable(tf.random.truncated_normal([multi_head*hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(multi_head*hidden_nums*output_nums))))

        
    def __call__(self,x,k=None,v=None):
        '''
        x:(batch_size,seq_size,embeded_size)
        '''
        Zs=[]
      
        Ks=[]

        Vs=[]
        
        
        #mask_matrix=tf.zeros([1,1])
        for i in range(self.multi_head):
            q_name="q"+str(i)
            k_name="k"+str(i)
            v_name="v"+str(i)
            Q=tf.matmul(x,self.variables[q_name])
            K=tf.matmul(x,self.variables[k_name])
            V=tf.matmul(x,self.variables[v_name])

            Ks.append(K)
            Vs.append(V)
            
            score=tf.matmul(Q,tf.transpose(K,[0,2,1]))
            score=score/(math.sqrt(self.input_nums))
            if self.mask == True:#构建mask矩阵
                matrix=[]
                for j in range(score.shape[-1]):
                    matrix.append(tf.concat([tf.ones([K.shape[0],1,j+1]),tf.ones([K.shape[0],1,score.shape[-1]-(j+1)])*(-1.0)*np.inf],2))
                
                mask_matrix=tf.concat(matrix,1)
                
                score=score+mask_matrix
                
            score=tf.nn.softmax(score,axis=2)
            
            z=tf.matmul(score,V)
            
            Zs.append(z)
        
        z_output=tf.concat(Zs,len(Zs[0].shape)-1)
        
        z_output=tf.matmul(z_output,self.w0)

        return z_output,Ks,Vs
    def get_q_w0_params(self):
        
        params=[]
        for i in range(self.multi_head):
            q_name="q"+str(i)
            params.append(self.variables[q_name])
        
        params.append(self.w0)
        
        return params

    def get_k_v_params(self):
        params=[]
        for i in range(self.multi_head):
            k_name="k"+str(i)
            v_name="v"+str(i)
            params.append(self.variables[k_name])
            params.append(self.variables[v_name])
        
        return params
        
    def get_params(self):
        params=list(self.variables.values())
        params.append(self.w0)
        return params
    
    def init(self):
        pass
        #self.t=0
        #self.step=0
    def t_plus_one(self):
        pass
        #self.t=self.t+1

class encoder_decoder_attention:
    def __init__(self,input_nums,hidden_nums,output_nums,mask=False,multi_head=1):
        '''
        function:
            初始化attention
        parameter:
            input_nums:一般是词向量大小
        '''
        self.multi_head=multi_head
        self.input_nums=input_nums
        self.mask=mask
        
        self.step=0
        self.t=0
        
        
        self.variables={}
        for i in range(multi_head):
            q_name="q"+str(i)
            self.variables[q_name]=tf.Variable(tf.random.truncated_normal([input_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(input_nums+hidden_nums))))

        self.w0=tf.Variable(tf.random.truncated_normal([multi_head*hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(multi_head*hidden_nums*output_nums))))
    def __call__(self,x,k,v):
        Zs=[]

        for i in range(len(self.variables)):
            q_name="q"+str(i)
            Q=tf.matmul(x,self.variables[q_name])
            
            score=tf.matmul(Q,tf.transpose(k[i],[0,2,1]))
            score=score/math.sqrt(self.input_nums)
            
                
                
           
            score=tf.nn.softmax(score,axis=1)
            
            z=tf.matmul(score,v[i])
            
            Zs.append(z)

        z_output=tf.concat(Zs,len(Zs[0].shape)-1)

        z_output=tf.matmul(z_output,self.w0)

        

        return z_output,k,v
        
    def get_q_w0_params(self):
        
        params=[]
        for i in range(self.multi_head):
            q_name="q"+str(i)
            params.append(self.variables[q_name])
        
        params.append(self.w0)
        
        return params

    def get_k_v_params(self):
        params=[]
        for i in range(self.multi_head):
            k_name="k"+str(i)
            v_name="v"+str(i)
            params.append(self.variables[k_name])
            params.append(self.variables[v_name])
        
        return params
    
    def get_params(self):
        params=list(self.variables.values())
        params.append(self.w0)
        return params

