class Config:
    def __init__(self):
        self.max_len = 64
        self.batch_size = 64
        self.epoch = 10
        self.train_path = './data/train.txt'
        self.val_path = './data/val.txt'
        self.test_path = './data/test.txt'
        self.lr = 2e-5
        self.eps = 1e-8
