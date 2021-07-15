# chinese-tokenization-bert
This was a Chinese tokenization which consist of bert.

这是个中文序列标注的一个例子。用bert模型来分词，用B和I来标注词语。 

这个模型测试准确率低，大可能是我在训练时，隐藏层大小只为4。如果需要进一步提高，可增加这个隐藏层大小和bert的层数。

直接运行train_bert_tokenize.py就能进行训练。  
运行test_bert_tokenize.py就能进行测试。 

测试准确率：74%  

关键库：
tensorflow 2.4

参考  
-
1.[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)  
2.[Chinese Named Entity Recognition for Social Media](https://github.com/hltcoe/golden-horse) 
