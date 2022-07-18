import logging

import config

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s, line:%(lineno)d - %(message)s" 带有文件名、行号
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 这里设置日志的级别

# 输出到文件
file_handler = logging.FileHandler(config.IRBOPP + "log_config/all.log", mode='a', encoding='utf-8')

# 输出到控制台
stream_handler = logging.StreamHandler()

# 错误日志单独输出到一个文件
error_handler = logging.FileHandler(config.IRBOPP + "log_config/error.log", mode='a', encoding='utf-8')
error_handler.setLevel(logging.ERROR)
# 注意这里，错误日志只记录ERROR级别的日志

# 将所有的处理器加入到logger中
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.addHandler(error_handler)

formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

# 设置格式化
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# logger.info('info级别的')
# logger.error('error级别')
# logger.debug('debug级别')
# logger.warning('warning级别')
