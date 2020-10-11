class Dependency(object):
    def __init__(self, sid: int,
                 form: str,
                 tag: str,
                 head: int,
                 dep_rel: str):
        self.id = sid       # 当前词ID
        self.form = form    # 当前词（或标点）
        self.pos_tag = tag      # 词性
        self.head = head    # 当前词的head (ROOT为0)
        self.dep_rel = dep_rel  # 当前词与head之间的(句法/语义)依存关系

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


