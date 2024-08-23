import logging
import os

OK = "\033[92m"
WARNING = "\033[93m"
FAIL = "\033[91m"
END = "\033[0m"

PINK = "\033[95m"
BLUE = "\033[94m"
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING


class Colorlogger(object):
    """
    这段代码定义了一个 Colorlogger 类，该类用于创建具有不同颜色日志的记录器。以下是该类的主要功能和属性：
        初始化方法 (__init__)：
            接收两个参数：log_dir（日志存储的目录）和 log_name（日志文件名，默认为 "train_logs.txt"）
            设置记录器的日志级别为 INFO。
            创建文件日志处理器，将日志写入指定的文件（路径为 log_dir/log_name）
            创建控制台日志处理器，将日志输出到控制台
            设置日志的格式，包括时间、日志级别和消息
            将两个处理器添加到记录器中
        日志级别方法：
            debug(msg): 记录 DEBUG 级别的消息。
            info(msg): 记录 INFO 级别的消息。
            warning(msg): 记录 WARNING 级别的消息，并使用黄色文字标记。
            critical(msg): 记录 CRITICAL 级别的消息，并使用黄色文字标记（此处和 warning 方法的实现相同）。
            error(msg): 记录 ERROR 级别的消息，并使用红色文字标记。
        颜色常量：
            定义了一些 ANSI 转义序列，用于在终端中显示不同颜色的文字。
        此记录器可以同时将日志信息写入文件和打印到控制台，并通过颜色标记以提高可读性。
    """
    def __init__(self, log_dir, log_name="train_logs.txt"):
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_log = logging.FileHandler(log_file, mode="a")
        file_log.setLevel(logging.INFO)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        formatter = logging.Formatter("{}%(asctime)s{} %(message)s".format(GREEN, END), "%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        console_log.setFormatter(formatter)
        self._logger.addHandler(file_log)
        self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + "WRN: " + str(msg) + END)

    def critical(self, msg):
        self._logger.warning(WARNING + "WRN: " + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + "ERR: " + str(msg) + END)
