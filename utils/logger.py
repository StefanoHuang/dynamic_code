import logging

class Log():
    def __init__(self, name):
        # 打印日志
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.StreamHandler()
        fmt = "%(asctime)-10s %(levelname)s %(message)s"
        datefmt = "%H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def getlog(self):
        return self.logger
