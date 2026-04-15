transformer 从文本中**提取信息 , 生成特征向量**(encoder) , 计算 , 得到推理后的特征向量, 再重新转化成文本(decoder)


le-wm不需要文本生成 , 拆掉decoder, 此时无法从文本输入训练 , 