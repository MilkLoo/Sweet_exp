### 多人2D姿态估计模型

**`功能:`** 模型可以用于图片，视频或者在线摄像设备。
(在这里我只是用来生成人体的 *2D* 关键点位置坐标，对之前没有实现的功能做一个补充)

**`说明:`** 关于在 `2d_psoe_transformer` 文件夹下的 `relation_trans` 文件是用于生成单张
图片的 *2D* 姿态信息的，并保存在 `2d_psoe_est.json` 文件中的，注意每次只能生成一张的，重新运行
代码会覆盖之前的，具体的文件注释有说明。整个代码已经调试好了，只需要图片数据和对位姿信息的汇总了。
