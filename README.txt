说话人确认（不能直接运行，需先下载SLR68或SLR38数据集进行预处理，默认是SLR68）
Step1: 控制台输入 python encoder_preprocess.py
Step2: 控制台输入 visdom
Step3: 控制台输入 python encoder_train.py（会自动打开网页并开始训练，每隔一定步数显示可视化更新）


语音合成（不能直接运行，须先下载标贝数据集进行预处理）
Step1: 控制台输入 python preprocess.py（预处理）
Step2: 控制台输入 python train.py（开始训练，生成的模型和图都在./logs-tacotron文件夹中）
Step3: 控制台输入 python eval.py（通过./logs-tacotron文件夹中最新的模型进行语音合成，合成的语音输出路径默认为./logs-tacotron文件夹）

注：因为已经在./logs-tacotron文件夹中保存了100000步时的model和checkpoint，只需在eval.py中修改"sentence=['...']"中的文本信息（输入汉字和注音字符均可），然后运行eval.py即可产生默认名称为eval-100000.wav的语音。

Git is a distributed version control system.
Git is free software distributed under the GPL.