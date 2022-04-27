import logging

# 创建logger对象
logger = logging.getLogger('test_logger')

# 设置日志等级
logger.setLevel(logging.DEBUG)

# 追加写入文件a ，设置utf-8编码防止中文写入乱码
test_log = logging.FileHandler('test.log', 'a', encoding='utf-8')

# 向文件输出的日志级别
test_log.setLevel(logging.DEBUG)

# 向文件输出的日志信息格式
formatter = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')

test_log.setFormatter(formatter)

# 加载文件到logger对象中
logger.addHandler(test_log)

if __name__ == "__main__":
    logger.debug('----调试信息 [debug]------')
    logger.info('[info]')
    logger.warning('警告信息[warning]')
    logger.error('错误信息[error]')
    logger.critical('严重错误信息[crtical]')