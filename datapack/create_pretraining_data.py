# Coding=utf-8
# 本代码基于 Google AI 的 BERT 预训练数据生成代码改编。
# 主体思想与原始代码相近，但是是基于中文文本特点与 Pytorch 环境进行改编的，
# 用于将中文无标注文本处理为 BERT 的 masked LM/next sentence 预训练语料。
import sys
sys.path.append(r"/home/cjr/similarity-model/")
import random
import click
from loguru import logger
import gzip
import jieba
from ernie.classification import tokenization
import traceback


class TrainingInstance(object):
    """ 句子对形式的单个训练数据实例类型 """

    def __init__(self, tokens, segment_ids, word_seg_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.word_seg_labels = word_seg_labels
        self.is_random_next = is_random_next

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([x for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "word_seg_labels: %s\n" % (" ".join([x for x in self.word_seg_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length, output_file):
    """ 以 TrainingInstance 按照 ernie 模型的输入格式将数据写入 gzip 文件"""
    try:
        fout_t = gzip.open(output_file + '.train.gz', 'wb')
        fout_e = gzip.open(output_file + '.eval.gz', 'wb')
        total_written = 0
        for instance in instances:
            input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
            sent_ids = list(instance.segment_ids)
            pos_ids = [i for i in range(len(input_ids))]
            seg_labels = list(instance.word_seg_labels)
            assert len(input_ids) <= max_seq_length

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                sent_ids.append(0)
                pos_ids.append(0)
                seg_labels.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(sent_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length
            assert len(seg_labels) == max_seq_length

            # 这里注意：按照原文思路，随机下一句的 label 为 1，而真实下一句的 label 为 0 。
            next_sentence_label = 1 if instance.is_random_next else 0

            line = create_line((input_ids, sent_ids, pos_ids, seg_labels)) + str(next_sentence_label) + '\n'

            if total_written < len(instances) * 0.10:
                fout_e.write(line.encode('utf8'))
            else:
                fout_t.write(line.encode('utf8'))

            total_written += 1

    except Exception as e:
        traceback.print_exc()
        logger.error(e)
    finally:
        fout_t.close()
        fout_e.close()

    logger.info("Wrote %d total instances" % total_written)


def create_line(pack):
    res = ''
    for ids in pack:
        res += ' '.join(str(id) for id in ids)
        res += ';'
    return res


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, rng):
    """ 从原始语料生成 TrainingInstance 类实例"""
    all_documents = [[]]

    # 输入语料的数据格式：
    # (1) 每行是一整句完整句子。
    # (2) 每个段落之间用一个空行隔开。
    for input_file in input_files:
        with open(input_file, "r") as reader:
            while True:
                # TODO 这里没有实现判断是否为 unicode 编码
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # 如果是个空行，则表示新的段落开始。
                if not line:
                    all_documents.append([])
                # 这里先不进行 tokenize 因为后续的步骤里面需要通过 jieba 进行分词，最后在进行 tokenize
                # 也就是这里的 tokens 实际上是一个原始句子 str
                tokens = line
                if tokens:
                    all_documents[-1].append(tokens)

    # 去除空段落
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob, tokenizer, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob, tokenizer, rng):
    """ 从一个文本段落生成 TrainingInstances 类型实例"""
    document = all_documents[document_index]

    # 因为句子包含 [CLS], [SEP], [SEP] 所以长度减三
    max_num_tokens = max_seq_length - 3

    # 有概率在满足 max_seq_length 的前提下随机决定新的句子总长，
    # 以增加数据的任意性。
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # 这里采取的策略是先预选中最大长度的几个句子片段，然后将片段以完整句子为单位随机分为 A B 两部分，
    # 如果是进入了随机下一句的 case ，则在选完 A 中的句子后，下一句从其他段落中随机挑选句子填满该 instance。
    # 
    # 新版：这里不再像之前一样直接拿上一层分词好的结果进行运算，而是直接操作原始的字符串了，所以以下的代码从以前的操作 token 列表变成了操作 str。
    # 因为 jieba 分词后 tokenizer 的 token 结果可能与原始句子的 token 不一致，所以不能再在上层调用里提前分好词了。
    # 我们现在要先用 jieba 分词，然后对分词后的 tokenize 结果生成 word_seg_labels 和 ids 并同步进行截断。
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # a_end 代表从 current_chunk 中选择多少个句子放入 A 片段
                # A 代表第一句话
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                str_a = ''
                for j in range(a_end):
                    str_a += current_chunk[j]

                str_b = ''
                # 随机的 “下一句话”
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(str_a)

                    # 随机选取一个其他的段落以摘取一个“任意的”下一句话
                    random_document_index = rng.randint(0, len(all_documents) - 1)
                    for _ in range(10):
                        if random_document_index != document_index:
                            break
                        random_document_index = rng.randint(0, len(all_documents) - 1)

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        str_b += random_document[j]
                        if len(str_b) >= target_b_length:
                            break
                    # 这里由于采取了随机的下一句话，所以原本想当作第二句话的句子都可以放回去等待下一轮使用。
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # 真实的 “下一句话” 
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        str_b += current_chunk[j]

                # 引入中文分词，这里简单采用 jieba 库进行分词。按照 ernie 的设计
                # word_seg_labels 表示分词边界信息: 0表示词首、1表示非词首、-1为占位符, 其对应的词为 CLS或者 SEP

                tokens_a = []
                segment_ids_a = []
                word_seg_labels_a = []
                
                words = list(jieba.cut(str_a))
                for word in words:
                    word_tokens = tokenizer.tokenize(word)
                    if len(word_tokens) > 0:
                        word_seg_labels_a += [0] + [1] * (len(word_tokens) - 1)
                        for token in word_tokens:
                            tokens_a.append(token)
                            segment_ids_a.append(0)
                        assert len(tokens_a) == len(word_seg_labels_a), "tokens_a: {}, word_seg_labels_a: {}, word: {}".format(
                            len(tokens_a), len(word_seg_labels_a), word)

                tokens_b = []
                segment_ids_b = []
                word_seg_labels_b = []
                words = list(jieba.cut(str_b))
                for word in words:
                    word_tokens = tokenizer.tokenize(word)
                    if len(word_tokens) > 0:
                        word_seg_labels_b += [0] + [1] * (len(word_tokens) - 1)
                        for token in word_tokens:
                            tokens_b.append(token)
                            segment_ids_b.append(1)
                        assert len(tokens_b) == len(word_seg_labels_b), "tokens_a: {}, word_seg_labels_a: {}, word: {}".format(
                            len(tokens_b), len(word_seg_labels_b), word)

                # 这里是同时对 A B 两个 turple 里的所以列表进行截断，保证最终的 instance 长度不超过 max_num_tokens
                truncate_seq_pair((tokens_a,segment_ids_a,word_seg_labels_a), (tokens_b,segment_ids_b,word_seg_labels_b), max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = ["[CLS]"]
                segment_ids = [0]
                word_seg_labels = [-1]

                tokens += tokens_a
                segment_ids += segment_ids_a
                word_seg_labels += word_seg_labels_a
                
                tokens.append("[SEP]")
                segment_ids.append(0)
                word_seg_labels.append(-1)

                tokens += tokens_b
                segment_ids += segment_ids_b
                word_seg_labels += word_seg_labels_b
                
                tokens.append("[SEP]")
                segment_ids.append(1)
                word_seg_labels.append(-1)

                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    word_seg_labels=word_seg_labels,
                    is_random_next=is_random_next)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def truncate_seq_pair(turple_a, turple_b, max_num_tokens, rng):
    """ 
        将两个句子修剪至总长度小于等于设定的最大长度 
        turple 里面装的是 tokens, segment_ids, word_seg_labels
    """
    while True:
        total_length = len(turple_a[0]) + len(turple_b[0])
        if total_length <= max_num_tokens:
            break

        trunc_turple = turple_a if len(turple_a[0]) > len(turple_b[0]) else turple_b
        assert len(trunc_turple[0]) >= 1

        trunc_turple[0].pop()
        trunc_turple[1].pop()
        trunc_turple[2].pop()


@click.command()
@click.option("--input_file", help="Input raw text file (or comma-separated list of files).")
@click.option("--output_file", help="Output TF example file (or comma-separated list of files).")
@click.option("--max_seq_length", default=128, help="Maximum sequence length.")
@click.option("--random_seed", default=519, help="Random seed for data generation.")
@click.option("--dupe_factor", default=10, help="Number of times to duplicate the input data (with different masks).")
@click.option("--short_seq_prob", default=0.1, type=float,
              help="Probability of creating sequences which are shorter than the maximum length.")
@click.option("--vocab_path", help="The vocabulary file that the Ernie model was trained on.")
@click.option("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text.")
def main(input_file, output_file, max_seq_length, random_seed, dupe_factor, short_seq_prob, vocab_path, do_lower_case):
    logger.info("*** Loading the tokenizer ***")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=do_lower_case)

    # TODO 这里没有实现原始的给定文件 pattern 来批量匹配文件
    input_files = input_file.split(",")
    logger.info("*** Reading from input files ***")
    for input_file in input_files:
        logger.info("  %s" % input_file)

    rng = random.Random(random_seed)
    instances = create_training_instances(input_files, tokenizer, max_seq_length, dupe_factor, short_seq_prob, rng)

    logger.info("*** Writing to output file ***")
    logger.info("  %s.train  %s.eval" % (output_file, output_file))

    write_instance_to_example_files(instances, tokenizer, max_seq_length, output_file)


if __name__ == "__main__":
    main()
