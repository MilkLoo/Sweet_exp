import os
import os.path as osp
import math
import time
import glob
import abc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from common.timer import Timer
from common.logger import Colorlogger
from torch.nn.parallel.data_parallel import DataParallel
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from config import cfg
from model import get_model
# from model_transformer import get_model
# from model_cnn_transformer import get_model
from dataloader.dataset import MultipleDatasets

"""
代码动态地从四个不同的模块中导入了同名的类，并将它们分别存储在 Human36M、MPII、MSCOCO 和 PW3D 变量中。
这种做法可以根据需要灵活地导入不同模块中的类，而不需要显式地写出每个导入语句
"""
dataset_list = ['Human36M', 'MSCOCO', "MuCo", "CrowdPose", "PW3D", "MPII", "OH3D"]  # 以后可以自己添加
for i in range(len(dataset_list)):
    exec('from ' + "dataloader." + dataset_list[i] + ' import ' + dataset_list[i])


class Base(object):
    """
    这是一个抽象基类 Base，它定义了一个抽象方法 _make_batch_generator 和 _make_model，这些方法在子类中必须被实现。
    Base 类还包含了一些共享的功能，比如初始化计时器（tot_timer、gpu_timer 和 read_timer）和日志记录器（logger）等。子类可以继承 Base 类，
    并根据需要实现抽象方法，以便为特定的任务创建批处理生成器和模型。
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = Colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    """
    除注释之外的
    Trainer 类继承自 Base 类，它实现了 _make_batch_generator 和 _make_model 方法，并且添加了一些额外的方法来管理训练过程
    ，如 get_optimizer、save_model、load_model、set_lr、get_lr。以下是 Trainer 类的主要功能：
        get_optimizer: 返回一个配置好的优化器，用于模型训练时的参数更新。
        save_model: 保存模型和优化器的状态。
        load_model: 加载模型和优化器的状态，以便从之前保存的检查点继续训练。
        set_lr: 根据当前训练的 epoch 设置学习率。
        get_lr: 获取当前学习率。
        _make_batch_generator: 创建用于产生训练数据批次的数据加载器。
        _make_model: 创建模型，并加载模型参数，如果需要，还会加载之前保存的检查点。
    此外，Trainer 类还包含了一些额外的属性，、
    例如 vertex_num、joint_num、itr_per_epoch、batch_generator、start_epoch、model 和 optimizer，用于在训练过程中跟踪状态。
    """

    """
    用了注释的
    这个 Trainer 类继承自 Base 类，用于模型的训练。下面是对该类进行的修改和添加：
        在 __init__ 方法中，通过传递参数 cfg 来初始化配置，并调用 Base 类的构造函数初始化日志记录器。
        get_optimizer 方法现在接受一个模型作为参数，并返回一个配置好的优化器。优化器现在根据模型的结构分成了不同的参数组，并使用AdamW优化器。
        save_model 方法现在还保存了额外的信息，并且只有在当前进程的 rank 为 0 时才会记录日志。
        load_model 方法在加载模型参数时，还会加载 awl 参数。
        set_lr 方法和 get_lr 方法现在使用了配置中的学习率调整参数。
        _make_batch_generator 方法现在创建数据加载器时，根据配置选择是否使用分布式数据并行加载。
        _make_model 方法在创建模型时，根据配置使用了分布式数据并行，同时初始化了自动加权损失函数 awl。
    """

    def __init__(self):
        # self.cfg = cfg
        # super(Trainer, self).__init__(cfg.log_dir, log_name='train_logs.txt')
        super(Trainer, self).__init__(log_name='train_logs.txt')

    # 这是第二个模型的
    def get_optimizer(self, model):
        all_params = model.parameters()
        # optimizer = torch.optim.AdamW([{'params': all_params}],
        #                               lr=cfg.lr, weight_decay=cfg.weight_decay)
        optimizer = torch.optim.AdamW([{'params': all_params}],lr=cfg.lr)
        print('The parameters of model are added to the optimizer !')
        if cfg.resume_ckpt != "":
            print("Load optimizer from: {}".format(cfg.resume_ckpt))
            model_path = cfg.resume_ckpt
            ckpt = torch.load(model_path)
            optimizer.load_state_dict(ckpt["optimizer"])
        return optimizer

    # def get_optimizer(self, model):
    #     base_params = list(map(id, model.module.backbone.parameters()))
    #     other_params = filter(lambda p: id(p) not in base_params, model.module.parameters())
    #     other_params = list(other_params)
    #     optimizer = torch.optim.AdamW([
    #         {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
    #         {'params': other_params},
    #         # {'params': self.awl.parameters(), 'weight_decay': 0}
    #     ],
    #         lr=cfg.lr, weight_decay=cfg.weight_decay
    #     )
    #     if cfg.resume_ckpt != "":
    #         print("Load optimizer from: {}".format(cfg.resume_ckpt))
    #         model_path = cfg.resume_ckpt
    #         ckpt = torch.load(model_path)
    #         optimizer.load_state_dict(ckpt["optimizer"])
    #     return optimizer

    #  改了网络这里也需要改

    # @staticmethod
    # def get_optimizer(model):
    #     optimizer = torch.optim.Adam([
    #         {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
    #         {'params': model.module.pose2feat.parameters()},
    #         {'params': model.module.position_net.parameters()},
    #         {'params': model.module.rotation_net.parameters()},
    #          加了模块这里也要写
    #         {'params': model.module.module.parameters()}
    #     ],
    #         lr=cfg.lr)
    #     print('The parameters of backbone, pose2feat, position_net, rotation_net, are added to the optimizer.')
    #
    #     return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        # if dist.get_rank() == 0:
        #    self.logger.info("Write snapshot into {}".format(file_path))
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        # ckpt = torch.load(ckpt_path, map_location='cpu')  # -->  这个应该不影响
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1

        # 这个是研究内容1 要更改的
        # model.load_state_dict(ckpt['network'], strict=False)
        # # optimizer.load_state_dict(ckpt['optimizer'])
        # self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        # return start_epoch, model, optimizer

        model.load_state_dict(ckpt['network'], strict=False)
        # if cur_epoch != 0:
        #     self.awl.load_state_dict(ckpt['awl'])
        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model.cuda(), optimizer

    def set_lr(self, epoch):
        global e
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    # def set_lr(self, epoch):
    #     for e in cfg.lr_dec_epoch:
    #         if epoch < e:
    #             break
    #     if epoch < cfg.lr_dec_epoch[-1]:
    #         idx = cfg.lr_dec_epoch.index(e)
    #         for g in self.optimizer.param_groups:
    #             g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
    #     else:
    #         for g in self.optimizer.param_groups:
    #             g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            """
            在 Python 中，eval() 函数可以将字符串中的表达式作为 Python 代码来执行，并返回表达式的结果。
            但是 eval() 函数不能直接将字符串转换为变量。如果字符串中是一个变量名，
            eval() 函数会尝试执行该变量名对应的表达式，并返回其结果。
            """
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))

        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus * cfg.train_batch_size,
                                          shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(self.vertex_num, self.joint_num, 'train')
        # 这是第二个模型的
        # awl = AutomaticWeightedLoss(7).cuda()
        model = DataParallel(model).cuda()
        # self.awl = awl
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
    # def _make_model(self):
    #     # prepare network
    #     self.logger.info("Creating graph and optimizer...")
    #     model = get_model(self.vertex_num, self.joint_num, 'train')
    #     # 这是第二个模型的
    #     # awl = AutomaticWeightedLoss(6).cuda()
    #     model = DataParallel(model).cuda()
    #     # self.awl = awl
    #     optimizer = self.get_optimizer(model)
    #     if cfg.continue_train:
    #         start_epoch, model, optimizer = self.load_model(model, optimizer)
    #     else:
    #         start_epoch = 0
    #     model.train()
    #
    #     self.start_epoch = start_epoch
    #     self.model = model
    #     self.optimizer = optimizer

    # def _make_model(self):
    #     # prepare network
    #     self.logger.info("Creating graph and optimizer...")
    #     model = get_model(self.vertex_num, self.joint_num, 'train')
    #     model = DataParallel(model).cuda()
    #     optimizer = self.get_optimizer(model)
    #     if cfg.continue_train:
    #         start_epoch, model, optimizer = self.load_model(model, optimizer)
    #     else:
    #         start_epoch = 0
    #     model.train()
    #
    #     self.start_epoch = start_epoch
    #     self.model = model
    #     self.optimizer = optimizer


class Tester(Base):
    """
    Tester 类是用于模型测试的类，也是从 Base 类继承而来的。下面是对该类进行的修改和添加：
        在 __init__ 方法中，通过传递参数 test_epoch 来初始化测试的轮次，并调用 Base 类的构造函数初始化日志记录器。
        _make_batch_generator 方法现在创建数据加载器时，根据配置选择是否使用分布式数据并行加载。
        _make_model 方法在加载模型时，根据指定的测试轮次从对应的路径加载模型参数，并使用 DataParallel 将模型移到 GPU 上，并设置为评估模式。
        _evaluate 方法现在接受模型输出和当前样本索引作为参数，并调用测试集对象的 evaluate 方法对模型输出进行评估。
        _print_eval_result 方法用于打印评估结果。
    """

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")
        # 加载数据集  mode == “test” -->  需要看一下加载数据集
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset_loader
        self.vertex_num = testset_loader.vertex_num
        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.vertex_num, self.joint_num, 'test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.testset.print_eval_result(eval_result)


"""
AutomaticWeightedLoss 类是一个自动加权多任务损失函数。下面是对该类的解释：
    __init__ 方法初始化了参数 num，表示损失的数量，默认为 21。参数 num 决定了要学习的参数数量。
    forward 方法用于计算损失，接受一个字典 loss_dict，其中包含每个任务的损失。
            该方法首先根据损失的数量初始化学习参数，并且根据每个任务的损失计算加权损失。
            加权损失由每个任务的损失以及对应的学习参数计算得出，其中学习参数表示对应任务的重要性。
            
AutomaticWeightedLoss 类的主要功能是根据每个任务的损失自动调整任务权重，以最小化总体损失。具体来说，它可以完成以下任务：
    自动权重调整：根据任务的重要性动态调整任务的损失权重。这样可以使得训练过程中不同任务的损失对最终模型的影响更加平衡。
    多任务损失计算：根据给定的多个任务的损失值，计算加权后的总体损失。这个总体损失可以作为训练模型时的优化目标。
    适用性广泛：由于其自动权重调整的特性，AutomaticWeightedLoss 类适用于多任务学习中的各种场景，例如图像分类、目标检测、语义分割等。
"""


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multitask loss
    Params：
        num: int，the number of loss
        x: multitask loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    第二个模型设置为 6 对 6 个损失进行自动加权
    """

    def __init__(self, num=21):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True, dtype=torch.float32)
        self.params = nn.Parameter(params)

    def forward(self, loss_dict):
        if not hasattr(self, 'keys'):
            self.keys = sorted(list(loss_dict.keys()))
        for i, key in enumerate(self.keys):
            loss_dict[key] = 0.5 / (self.params[i] ** 2) * loss_dict[key] + torch.log(1 + self.params[i] ** 2)
        return loss_dict
