import time
import torch
import torch.nn.functional as F
import random
import numpy as np
from modules.optimizer import Optimizer
from model.sdparser import SDParser
from config.conf import args_config, data_config
from utils.dataset import DataLoader
from utils.datautil import load_data, create_vocab, batch_variable
import torch.nn.utils as nn_utils
from logger.logger import logger


class Trainer(object):
    def __init__(self, args, data_config):
        self.args = args
        self.data_config = data_config

        domain = args.data_type
        self.train_set, self.val_set, self.test_set = self.build_dataset(data_config, domain)
        self.vocabs = self.build_vocabs(data_config[domain]['train'], data_config['pretrained']['word_embedding'])

        self.model = SDParser(num_wds=len(self.vocabs['word']),
                              num_chars=len(self.vocabs['char']),
                              num_tags=len(self.vocabs['tag']),
                              wd_embed_dim=args.wd_embed_dim,
                              char_embed_dim=args.char_embed_dim,
                              tag_embed_dim=args.tag_embed_dim,
                              hidden_size=args.hidden_size,
                              num_layers=args.rnn_depth,
                              arc_size=args.arc_size,
                              rel_size=args.rel_size,
                              num_lbl=len(self.vocabs['rel']),
                              arc_drop=args.arc_drop,
                              rel_drop=args.rel_drop,
                              dropout=args.dropout,
                              embed_weight=self.vocabs['word'].embeddings).to(args.device)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %d trainable parameters..." % total_params)

    def build_dataset(self, data_config, domain='news'):
        train_set = load_data(data_config[domain]['train'])
        val_set = load_data(data_config[domain]['dev'])
        test_set = load_data(data_config[domain]['test'])
        print('train data size:', len(train_set))
        print('validate data size:', len(val_set))
        print('test data size:', len(test_set))
        return train_set, val_set, test_set

    def build_vocabs(self, train_data_path, embed_file=None):
        vocabs = create_vocab(train_data_path)
        embed_count = vocabs['word'].load_embeddings(embed_file)
        print("%d word pre-trained embeddings loaded..." % embed_count)
        # save_to(self.args.vocab_chkp, vocabs)
        return vocabs

    def calc_loss(self, head_score, rel_score, head_tgt, rel_tgt, mask=None):
        '''
        :param head_score: (b, t, t)
        :param rel_score: (b, t, t, c)
        :param head_tgt: (b, t, t)
        :param rel_tgt: (b, t, t)
        :param mask: (b, t)  1对应有效部分，0对应pad填充
        :return:
        '''
        bz, seq_len, seq_len, num_lbl = rel_score.data.size()
        pad_mask = mask.eq(0)
        # head_tgt: (bz, seq_len, seq_len)
        weights = torch.ones((bz, seq_len, seq_len), dtype=torch.float, device=head_score.device)
        weights = weights.masked_fill(pad_mask.unsqueeze(1), 0)
        weights = weights.masked_fill(pad_mask.unsqueeze(2), 0)
        head_loss = F.binary_cross_entropy_with_logits(head_score, head_tgt.float(), weight=weights, reduction='sum')

        # rel_tgt: (bz, seq_len, seq_len)
        rel_mask = rel_tgt.eq(0)
        rel_tgt = rel_tgt.masked_fill(rel_mask, -1)
        rel_score = rel_score.reshape(-1, num_lbl)
        rel_loss = F.cross_entropy(rel_score, rel_tgt.reshape(-1), ignore_index=-1, reduction='sum')
        loss = (head_loss + rel_loss) / mask.sum()  # num of words
        return loss

    def calc_acc(self, head_score, rel_score, head_tgt, rel_tgt):
        '''
        :param head_score: (b, t, t)
        :param rel_score: (b, t, t, c)
        :param head_tgt: (b, t, t)
        :param rel_tgt: (b, t, t)
        # :param mask: (b, t)  1对应有效部分，0对应pad填充
        :return:
        '''
        if float(torch.__version__[:3]) >= 1.2:
            head_tgt = head_tgt.bool()
        else:
            head_tgt = head_tgt.byte()

        pred_heads = head_score.data.sigmoid() >= 0.5
        head_acc = (head_tgt * pred_heads).sum().item()
        total_head = head_tgt.sum().item()

        pred_rels = rel_score.data.argmax(dim=3)  # (bs, seq_len, seq_len)
        rel_acc = (head_tgt * pred_heads * (pred_rels == rel_tgt)).sum().item()
        return head_acc, rel_acc, total_head

    def calc_prf(self, num_correct, num_pred, num_gold):
        p = num_correct / (num_pred + 1e-30)
        r = num_correct / (num_gold + 1e-30)
        f = (2 * num_correct) / (num_gold + num_pred)
        return p, r, f

    # def parse_pred_graph(self, graph_pred, lens, rel_vocab=None):
    #     '''
    #     :param graph_pred: (b, t, t)
    #     :param rel_vocab:
    #     :param lens: (b, )  每句话的实际长度
    #     :return:  [(wid, head_id, rel_id), ...]
    #     '''
    #     graph_pred = graph_pred.tolist()
    #     res = []
    #     for s, l in zip(graph_pred, lens):
    #         idx = 1
    #         arcs = []
    #         for w in s[1:l]:
    #             arc = []
    #             for head_idx, rel_idx in enumerate(w[:l]):
    #                 if rel_idx == 0:
    #                     continue
    #
    #                 if rel_vocab is None:
    #                     arc.append((idx, head_idx, rel_idx - 1))
    #                 else:
    #                     arc.append((idx, head_idx, rel_vocab.idx2inst(rel_idx - 1)))
    #             idx += 1
    #             arcs.extend(arc)
    #         res.append(arcs)
    #     return res
    #
    # def parse_gold_graph(self, rel_tgt, lens, rel_vocab=None):
    #     '''
    #     :param rel_tgt: (b, t, t)  head位置对应关系类型索引
    #     :param lens: (b, )  句子实际长度
    #     :param rel_vocab:
    #     :return: [(wid, head_id, rel_id), ...]
    #     '''
    #     gold_res = []
    #     rel_tgt = rel_tgt.tolist()
    #     for rel_ids, l in zip(rel_tgt, lens):
    #         wid = 1
    #         res = []
    #         for rel_id in rel_ids[1:l]:
    #             arc = []
    #             for hid, rid in enumerate(rel_id[:l]):
    #                 if rid == 0:
    #                     continue
    #
    #                 if rel_vocab is None:
    #                     arc.append((wid, hid, rid))
    #                 else:
    #                     arc.append((wid, hid, rel_vocab.idx2inst(rid)))
    #             wid += 1
    #             res.extend(arc)
    #         gold_res.append(res)
    #     return gold_res

    def parse_pred_graph(self, graph_pred, lens, rel_vocab=None):
        '''
        :param graph_pred: (b, t, t)
        :param rel_vocab:
        :param lens: (b, )  每句话的实际长度
        :return:  [(wid, head_id, rel_id), ...]
        '''
        graph_pred = graph_pred.tolist()
        res = []
        for s, l in zip(graph_pred, lens):
            arcs = []
            for w in s[1:l]:
                arc = []
                for head_idx, rel_idx in enumerate(w[:l]):
                    if rel_idx == 0:
                        continue

                    if rel_vocab is None:
                        arc.append((head_idx, rel_idx - 1))
                    else:
                        arc.append((head_idx, rel_vocab.idx2inst(rel_idx - 1)))
                arcs.append(arc)
            res.append(arcs)
        return res

    def parse_gold_graph(self, rel_tgt, lens, rel_vocab=None):
        '''
        :param rel_tgt: (b, t, t)  head位置对应关系类型索引
        :param lens: (b, )  句子实际长度
        :param rel_vocab:
        :return:
        '''
        gold_res = []
        rel_tgt = rel_tgt.tolist()
        for rel_ids, l in zip(rel_tgt, lens):
            res = []
            for rel_id in rel_ids[1:l]:
                arc = []
                for hid, rid in enumerate(rel_id[:l]):
                    if rid == 0:
                        continue

                    if rel_vocab is None:
                        arc.append((hid, rid))
                    else:
                        arc.append((hid, rel_vocab.idx2inst(rid)))
                res.append(arc)
            gold_res.append(res)
        return gold_res

    def train_eval(self):
        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)
        self.args.max_step = self.args.epoch * (len(train_loader) // self.args.update_step)
        print('max step:', self.args.max_step)
        optimizer = Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), args)
        best_dev_metric, best_test_metric = dict(), dict()
        patient = 0
        for ep in range(1, 1+self.args.epoch):
            train_loss = 0.
            self.model.train()
            t1 = time.time()
            train_head_acc, train_rel_acc, train_total_head = 0, 0, 0
            for i, batcher in enumerate(train_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)

                head_score, rel_score = self.model(batch.wd_ids, batch.ch_ids, batch.tag_ids)
                loss = self.calc_loss(head_score, rel_score, batch.head_ids, batch.rel_ids, batch.wd_ids.gt(0))
                loss_val = loss.data.item()
                train_loss += loss_val

                head_acc, rel_acc, total_head = self.calc_acc(head_score, rel_score, batch.head_ids, batch.rel_ids)
                train_head_acc += head_acc
                train_rel_acc += rel_acc
                train_total_head += total_head

                if self.args.update_step > 1:
                    loss = loss / self.args.update_step

                loss.backward()

                if (i + 1) % self.args.update_step == 0 or (i == self.args.max_step - 1):
                    nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                             max_norm=self.args.grad_clip)
                    optimizer.step()
                    self.model.zero_grad()

                logger.info('[Epoch %d] Iter%d time cost: %.2fs, lr: %.6f, train loss: %.3f, head acc: %.3f, rel acc: %.3f' % (
                    ep, i + 1, (time.time() - t1), optimizer.get_lr(), loss_val, train_head_acc/train_total_head, train_rel_acc/train_total_head))

            dev_metric = self.evaluate('dev')
            if dev_metric['uf'] > best_dev_metric.get('uf', 0):
                best_dev_metric = dev_metric
                test_metric = self.evaluate('test')
                if test_metric['uf'] > best_test_metric.get('uf', 0):
                    # check_point = {'model': self.model.state_dict(), 'settings': args}
                    # torch.save(check_point, self.args.model_chkp)
                    best_test_metric = test_metric
                patient = 0
            else:
                patient += 1

            logger.info('[Epoch %d] train loss: %.4f, lr: %f, patient: %d, dev_metric: %s, test_metric: %s' % (
                ep, train_loss, optimizer.get_lr(), patient, best_dev_metric, best_test_metric))

            # if patient == (self.args.patient // 2 + 1):  # 训练一定epoch, dev性能不上升, decay lr
            #     optimizer.lr_decay(0.95)

            if patient >= self.args.patient:  # early stopping
                break

        logger.info('Final Metric: %s' % best_test_metric)

    def evaluate(self, mode='test'):
        if mode == 'dev':
            test_loader = DataLoader(self.val_set, batch_size=self.args.test_batch_size)
        elif mode == 'test':
            test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)
        else:
            raise ValueError('Invalid Mode!!!')

        self.model.eval()
        rel_vocab = self.vocabs['rel']
        nb_head_gold, nb_head_pred, nb_head_correct = 0, 0, 0
        nb_rel_gold, nb_rel_pred, nb_rel_correct = 0, 0, 0
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.vocabs)
                batch.to_device(self.args.device)

                head_score, rel_score = self.model(batch.wd_ids, batch.ch_ids, batch.tag_ids)
                mask = batch.wd_ids.gt(0)
                lens = mask.sum(dim=1)
                graph_pred = self.model.graph_decode(head_score, rel_score, mask)

                pred_deps = self.parse_pred_graph(graph_pred, lens, rel_vocab)
                gold_deps = self.parse_gold_graph(batch.rel_ids, lens, rel_vocab)
                assert len(pred_deps) == len(gold_deps)
                # for deps_p, deps_g in zip(pred_deps, gold_deps):
                #     nb_head_gold += len(deps_g)
                #     nb_rel_gold += len(deps_g)
                #
                #     nb_head_pred += len(deps_p)
                #     nb_rel_pred += len(deps_p)
                #     for dg in deps_g:
                #         for dp in deps_p:
                #             if dg[:-1] == dp[:-1]:
                #                 nb_head_correct += 1
                #                 if dg == dp:
                #                     nb_rel_correct += 1
                #                 break

                for pdeps, gdeps in zip(pred_deps, gold_deps):  # sentence
                    assert len(pdeps) == len(gdeps)
                    for pdep, gdep in zip(pdeps, gdeps):  # word
                        nb_head_pred += len(pdep)
                        nb_rel_pred += len(pdep)

                        nb_head_gold += len(gdep)
                        nb_rel_gold += len(gdep)
                        for gd in gdep:  # (head_id, rel_id)
                            for pd in pdep:
                                if pd[0] == gd[0]:
                                    nb_head_correct += 1
                                    if pd == gd:
                                        nb_rel_correct += 1
                                    break

        up, ur, uf = self.calc_prf(nb_head_correct, nb_head_pred, nb_head_gold)
        lp, lr, lf = self.calc_prf(nb_rel_correct, nb_rel_pred, nb_rel_gold)
        return dict(
            up=up,
            ur=ur,
            uf=uf,
            lp=lp,
            lr=lr,
            lf=lf
        )


if __name__ == '__main__':
    random.seed(1347)
    np.random.seed(2343)
    torch.manual_seed(1453)
    torch.cuda.manual_seed(1347)
    torch.cuda.manual_seed_all(1453)

    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = data_config('./config/data_path.json')
    trainer = Trainer(args, data_path)
    trainer.train_eval()


