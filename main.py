from dataset import TrainDataset, EvalDataset
from model import BERT
from process import Trainer
from args import args
import torch.utils.data as Data


def main():
    train_dataset = TrainDataset(args.mask_prob, args.max_len)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    val_dataset = EvalDataset(args.max_len, args.sample_size, mode='val', enable_sample=args.enable_sample)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.val_batch_size)

    test_dataset = EvalDataset(args.max_len, args.sample_size, mode='test', enable_sample=args.enable_sample)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')

    model = BERT(args)
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    main()
