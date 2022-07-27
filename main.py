from train import train
from config import Config
from model import model

if __name__ == '__main__':
    config = Config()
    train(model, config)