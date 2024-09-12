import logging
import sys


def get_logger(logger_name = None):
    # 获取已配置的 logger
    # print(f'logger_name: {logger_name}')
    return logging.getLogger(logger_name)

def configure_logger(conf):
    logger_name = conf['logger']['logger_name']
    level = conf['logger']['level']

    log_level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    # 创建一个日志器logger并设置其日志级别为DEBUG
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level_mapping[level])  # DEBUG、INFO、WARNING、ERROR、CRITICAL

    # 创建一个流处理器handler并设置其日志级别为DEBUG
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level_mapping[level])

    # 创建一个格式器formatter并将其添加到处理器handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # 为日志器logger添加上面创建的处理器handler
    logger.addHandler(handler)

    # 日志输出
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
    return logger_name


if __name__ == '__main__':
    conf = {
        'logger' : {'logger_name':'FL-logger','level':'INFO'},
    }
    logger_name = configure_logger(conf)
    get_logger(logger_name=logger_name)
