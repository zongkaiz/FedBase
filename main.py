
from args import input_options
from src.fed_server.fedavg import FedAvgTrainer






def main():
    trainer= input_options()
    trainer.train()
if __name__ == '__main__':
    main()

