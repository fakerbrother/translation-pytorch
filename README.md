# translation-pytorch
## 本文参考自Pytorch的官网例子，链接为：https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 
## 代码解释
### 1 DataSet.py 读取数据，并将数据转换为torch.tensor()，数据地址为：http://www.manythings.org/anki/, 下载中英文翻译，作者将数据名称改为了en-cn.txt。
### 2 seq2seq_model.py seq2seq网络结构，包括encoder，decoder，decoder with attention。
### 3 train.py 训练代码。
### 4 eval_model.py 加载已经保存好的模型，并进行测试。测试时，由于没有进行数据预处理，所以输入的句子要保证单词在数据集里，并且要小写。
