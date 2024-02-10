# 基于bert-mini和MIND数据集的新闻推荐系统
## Requirements - 环境配置

`pip install -r requirements.txt`
`数据集：MINDlarge`
`基础模型：bert-mini`

## Usage - 训练
```
python main.py #训练代码，需要先下载MIND数据集和bert-mini并放入上一级目录
```
## Usage - 测试
```
python test.py --s1=0 --e1=10 #生成测试数据集第s1条到第e1条的结果
```
```
注：需要更改utils.py中MIND测试数据集目录如下
```
```python
news = pd.read_csv("../MIND/MINDlarge_test/news.tsv", delimiter='\s*\t\s*',header=None,index_col=0)
```
## Others - 其他
`bert-news.pth和user.pth是已训练完成的模型`
`GPU.py可用来观察GPU显存使用情况`

## Result - 结果
|  训练轮数   | AUC  |
|  ----  | ----  |
| 0.5% Epoch  | 0.58 |
| 2/3 Epoch | 0.65 |

## License - 开源协议
[MIT](https://choosealicense.com/licenses/mit/)
