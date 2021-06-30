import time
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer():
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        self.args = args
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(torch.device(self.device))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # lr_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

        self.num_epoch = args.num_epoch
        self.metric_ks = args.metric_ks
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.result_file = None

        self.step = 0
        self.best_metric = -1e9

    def train(self):
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            print('epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.compute_loss(batch)
            loss_sum += loss.item()
            loss.backward()

            self.optimizer.step()

            self.step += 1
            if self.step % self.eval_per_steps == 0:
                metric = {}
                for mode in ['val', 'test']:
                    metric[mode] = self.eval_model(mode)
                print(metric)
                if metric['test']['NDCG@20'] > self.best_metric:
                    torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                    self.result_file = open(self.save_path + '/result.txt', 'w+')
                    print('saving model of step{0}'.format(self.step), file=self.result_file)
                    print(metric, file=self.result_file)
                    self.result_file.close()
                    self.best_metric = metric['test']['NDCG@20']

        return loss_sum / idx, time.perf_counter() - t0

    def compute_loss(self, batch):
        seqs, labels = batch

        outputs = self.model(seqs)  # B * L * N
        outputs = outputs.view(-1, outputs.shape[-1])  # (B*L) * N
        labels = labels.view(-1)

        loss = self.ce(outputs, labels)
        return loss

    def eval_model(self, mode):
        self.model.eval()
        tqdm_data_loader = tqdm(self.val_loader) if mode == 'val' else tqdm(self.test_loader)
        metrics = {}

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]

                metrics_batch = self.compute_metrics(batch)

                for k, v in metrics_batch.items():
                    if not metrics.__contains__(k):
                        metrics[k] = v
                    else:
                        metrics[k] += v

        for k, v in metrics.items():
            metrics[k] = v / idx
        return metrics

    def compute_metrics(self, batch):
        seqs, answers, labels = batch
        scores = self.model(seqs)
        scores = scores[:, -1, :]  # the prediction score of the last `mask` token in seq, B * N
        scores = scores.gather(1, answers)  # only consider positive and negative items' score

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics
