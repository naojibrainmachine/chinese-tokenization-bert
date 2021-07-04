import tensorflow as tf

def layer_norm(z,axis=-1,eps=1e-9):
    '''
    z(batch_size,seq_size,hidden_size)
    '''
    mean=tf.reduce_mean(z,axis=axis,keepdims=True)
    std=tf.math.reduce_std(z,axis=axis,keepdims=True)
    
    return ((z-mean)/tf.math.sqrt(std**2+eps))

def add(x,y):
    return x+y

