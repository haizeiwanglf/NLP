# wandb使用
清华大学数据集文本分类。

对比

|      | base      | lora                 | prompt-tuning | p-tuning | p-tuning2 | prefix-tuning |
| ---- | --------- | -------------------- | ------------- | -------- | --------- | ------------- |
| acc  |           | 91.14                |               |          |           |               |
| p    |           | 91.15                |               |          |           |               |
| r    |           | 91.11                |               |          |           |               |
| f1   |           | 91.11                |               |          |           |               |
| 显存 | 4.3G      | 2.7G(后面上升到3.5G) |               |          |           |               |
| 时间 | 42.14分钟 | 30.01分钟            |               |          |           |               |
| 参数 | bert 1亿  | 362602（29.5%）      |               |          |           |               |

损失函数:

base

![image-20230528201632118](C:\Users\wushuo\AppData\Roaming\Typora\typora-user-images\image-20230528201632118.png)

lora

![image-20230528203301648](C:\Users\wushuo\AppData\Roaming\Typora\typora-user-images\image-20230528203301648.png)
