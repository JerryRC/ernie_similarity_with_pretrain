# Coding=utf-8
# 本代码基于 Google AI 的 BERT 预训练数据生成代码改编。
# 主体思想与原始代码相近，但是是基于中文文本特点与 Pytorch 环境进行改编的，
# 用于将中文无标注文本处理为 BERT 的 masked LM/next sentence 预训练语料。
import random
import click
from pytorch_transformers import BertTokenizer
from loguru import logger
import gzip
import jieba


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
        e.with_traceback()
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
                tokens = [line]
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

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # 随机的 “下一句话”
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # 随机选取一个其他的段落以摘取一个“任意的”下一句话
                    random_document_index = rng.randint(0, len(all_documents) - 1)
                    for _ in range(10):
                        if random_document_index != document_index:
                            break
                        random_document_index = rng.randint(0, len(all_documents) - 1)

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # 这里由于采取了随机的下一句话，所以原本想当作第二句话的句子都可以放回去等待下一轮使用。
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # 真实的 “下一句话” 
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # 引入中文分词，这里简单采用 jieba 库进行分词。按照 ernie 的设计
                # word_seg_labels 表示分词边界信息: 0表示词首、1表示非词首、-1为占位符, 其对应的词为 CLS或者 SEP

                tokens = []
                segment_ids = []
                word_seg_labels = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                word_seg_labels.append(-1)
                
                words = list(jieba.cut(''.join(i for i in tokens_a)))
                for word in words:
                    word_tokens = tokenizer.tokenize(word)
                    word_seg_labels += [0] + [1] * (len(word_tokens) - 1)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)
                word_seg_labels.append(-1)

                words = list(jieba.cut(''.join(i for i in tokens_b)))
                for word in words:
                    word_tokens = tokenizer.tokenize(word)
                    word_seg_labels += [0] + [1] * (len(word_tokens) - 1)
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

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


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """ 将两个句子修剪至总长度小于等于设定的最大长度 """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # 以二分之一的概率随机选择从句首或句尾截短当前句子
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


@click.command()
@click.option("--input_file", help="Input raw text file (or comma-separated list of files).")
@click.option("--output_file", help="Output TF example file (or comma-separated list of files).")
@click.option("--max_seq_length", default=128, help="Maximum sequence length.")
@click.option("--random_seed", default=519, help="Random seed for data generation.")
@click.option("--dupe_factor", default=10, help="Number of times to duplicate the input data (with different masks).")
@click.option("--short_seq_prob", default=0.1, type=float,
              help="Probability of creating sequences which are shorter than the maximum length.")
def main(input_file, output_file, max_seq_length, random_seed, dupe_factor, short_seq_prob):
    logger.info("*** Loading the tokenizer ***")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

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
