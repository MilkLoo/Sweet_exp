import time


class Timer(object):
    """
    A  simple timer
    提供的代码定义了一个名为Timer的简单计时器类。这个类旨在测量“tic”和“toc”操作之间经过的时间，提供了测量代码执行时间的功能。
    属性：
        total_time：在toc和tic调用之间累积的总时间。
        calls：计时器调用的次数。
        start_time：调用tic方法时记录的时间。
        diff：toc和tic调用之间的时间差。
        average_time：每次调用的平均时间（不包括热身阶段）。
        warm_up：计数器，用于处理热身调用，允许计时器稳定。
    方法：
        tic()：启动计时器。记录当前时间。
        toc(average=True)：停止计时器并计算自上次tic以来经过的时间。如果average为True，则返回每次调用的平均时间；否则返回最后一次调用的时间。
    计时器在前10次调用中包含一个热身阶段，以避免计时测量的潜在不稳定性。
    """

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.warm_up = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        if self.warm_up < 10:
            self.warm_up += 1
            return self.diff
        else:
            self.total_time += self.diff
            self.calls += 1
            self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff
