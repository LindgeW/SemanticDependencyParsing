import os
from utils.instance import Dependency
from utils.vocab import Vocab, MultiVocab
import torch


def read_deps(file_reader):
    deps = []
    for line in file_reader:
        try:
            tokens = line.strip().split('\t')
            if line.strip() == '' or len(tokens) < 10:
                if len(deps) > 0:
                    yield deps
                deps = []
            elif len(tokens) == 10:
                if tokens[6] == '_':
                    tokens[6] = '-1'
                deps.append(Dependency(int(tokens[0]), tokens[1], tokens[3], int(tokens[6]), tokens[7]))
        except Exception as e:
            print('exception occur: ', e)

    if len(deps) > 0:
        yield deps


def load_data(path):
    assert os.path.exists(path)
    dataset = []
    with open(path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr):
            dataset.append(deps)
    return dataset


def create_vocab(data_path, min_count=3):
    root_rel = None
    wd_vocab = Vocab(min_count, eos=None)
    char_vocab = Vocab(min_count, eos=None)
    tag_vocab = Vocab(eos=None)
    rel_vocab = Vocab(bos=None, eos=None)
    with open(data_path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr):
            for dep in deps:
                wd_vocab.add(dep.form)
                char_vocab.add(list(dep.form))
                tag_vocab.add(dep.pos_tag)

                if dep.head != 0:
                    rel_vocab.add(dep.dep_rel)
                elif root_rel is None:
                    root_rel = dep.dep_rel
                    rel_vocab.add(dep.dep_rel)
                elif root_rel != dep.dep_rel:
                    print('root = ' + root_rel + ', rel for root = ' + dep.dep_rel)

    return MultiVocab(dict(
        word=wd_vocab,
        char=char_vocab,
        tag=tag_vocab,
        rel=rel_vocab
    ))


def make_arc_rel_tgt(arcs, max_len, type_='arc'):
    assert type_ in ['arc', 'rel']
    batch_size = len(arcs)
    graph = torch.zeros(batch_size, max_len, max_len)  # 包含<root>节点
    for sent_idx, sent in enumerate(arcs):
        word_idx = 1
        # 每个词可能对应多个head和dep_rel
        for word in sent:  # ((head_idx1, rel_idx1), (head_idx2, rel_idx2), ..)
            for arc in word:  # (head_idx, rel_idx)
                head_idx = arc[0]
                idx = 1 if type_ == 'arc' else arc[1]
                graph[sent_idx, word_idx, head_idx] = idx
            word_idx += 1

    if type_ == 'arc':
        graph = graph.float()
    else:
        graph = graph.long()

    return graph


def batch_variable(batch_data, mVocab):
    batch_size = len(batch_data)
    max_seq_len = max(len(set(dep.id for dep in deps)) for deps in batch_data) + 1
    max_wd_len = max(len(dep.form) for deps in batch_data for dep in deps)

    wd_vocab = mVocab['word']
    ch_vocab = mVocab['char']
    tag_vocab = mVocab['tag']
    rel_vocab = mVocab['rel']

    wd_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    ch_ids = torch.zeros((batch_size, max_seq_len, max_wd_len), dtype=torch.long)
    tag_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    head_ids = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long)
    rel_ids = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.long)

    for i, deps in enumerate(batch_data):
        wds = []
        tags = []
        arc_dict = dict()
        for dep in deps:
            if dep.id not in arc_dict:
                wds.append(dep.form)
                tags.append(dep.pos_tag)
                arc_dict[dep.id] = [(dep.head, rel_vocab.inst2idx(dep.dep_rel))]
            else:
                arc_dict[dep.id].append((dep.head, rel_vocab.inst2idx(dep.dep_rel)))

        seq_len = len(wds) + 1
        wd_ids[i, :seq_len] = torch.tensor([wd_vocab.bos_idx] + wd_vocab.inst2idx(wds))
        tag_ids[i, :seq_len] = torch.tensor([tag_vocab.bos_idx] + tag_vocab.inst2idx(tags))

        for j, wd in enumerate(wds):
            if j == 0:
                ch_ids[i, j] = ch_vocab.bos_idx
            ch_ids[i, j+1, :len(wd)] = torch.tensor(ch_vocab.inst2idx(list(wd)))

        for wid, arcs in arc_dict.items():
            for head, rel in arcs:
                head_ids[i, wid, head] = 1
                rel_ids[i, wid, head] = rel

    return Batch(wd_ids, ch_ids, tag_ids, head_ids, rel_ids)


class Batch:
    def __init__(self, wd_ids, ch_ids, tag_ids, head_ids, rel_ids):
        self.wd_ids = wd_ids
        self.ch_ids = ch_ids
        self.tag_ids = tag_ids
        self.head_ids = head_ids
        self.rel_ids = rel_ids

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            setattr(self, prop, val.to(device))
        return self
