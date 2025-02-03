# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import regex as re
import requests
from tqdm import tqdm
from functools import lru_cache

'''
实现方式：
LRU 通常通过一个双向链表和哈希表来实现：
双向链表：用于记录数据的访问顺序。最近访问的数据放在链表头部，最久未访问的数据放在链表尾部。
哈希表：用于快速查找数据是否存在缓存中。
当访问某个数据时：
如果数据在缓存中，将其移动到链表头部（表示最近使用）。
如果数据不在缓存中，将其加入链表头部。如果缓存已满，则移除链表尾部的数据（最久未使用）
'''
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        # 检查缓存中是否已有该token的处理结果，有则直接返回
        if token in self.cache:
            return self.cache[token]
        # 将token转换为元组形式，便于后续处理
        word = tuple(token)
        # 获取token中的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回原token
        if not pairs:
            return token

        # 无限循环，直到无法进一步合并字符对
        while True:
            # 找到排名最低的字符对，即最应该合并的字符对
            # TODO 为什么找排名最低（频率最高）的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 如果该字符对不在排名中，跳出循环
            if bigram not in self.bpe_ranks:
                break
            # 获取字符对中的两个字符
            first, second = bigram
            # 初始化新的单词列表
            new_word = []
            i = 0
            # 遍历原单词中的字符
            while i < len(word):
                try:
                    # 找到第一个字符在原单词中的位置
                    j = word.index(first, i)
                    # 将位置之前的字符添加到新单词列表中
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 如果找不到第一个字符，将剩余字符添加到新单词列表中
                    new_word.extend(word[i:])
                    break

                # 如果当前位置的字符是第一个字符，且下一个字符是第二个字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 将这两个字符合并并添加到新单词列表中
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则，将当前字符添加到新单词列表中
                    new_word.append(word[i])
                    i += 1
            # 将新单词列表转换为元组形式
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
                # 否则，获取新单词中的所有字符对
        word = ' '.join(word)
        # 将最终处理后的单词转换为字符串形式
        self.cache[token] = word
        # 将结果缓存
        return word

    def encode(self, text):
        # 初始化一个空列表，用于存储BPE（Byte Pair Encoding）分词后的结果
        bpe_tokens = []
        # 使用正则表达式self.pat从输入文本text中找到所有匹配的子串
        for token in re.findall(self.pat, text):
            # 将找到的每个子串token编码为UTF-8字节序列
            # 然后通过self.byte_encoder将每个字节转换为对应的编码字符串
            # 最后将这些编码字符串连接成一个完整的字符串
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 使用BPE算法对编码后的字符串进行分词，得到BPE分词结果
            # 将分词结果按空格分割成多个子词
            # 然后通过self.encoder将每个BPE子词转换为对应的编码
            # 最后将这些编码扩展到bpe_tokens列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        # 返回最终的BPE分词编码结果列表
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # Modified code from
    subdir = 'gpt2_model'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


if __name__ == '__main__':
    text = "Hello, world. Is this-- a test?"
    orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir="ch02/02_bonus_bytepair-encoder")
    integers = orig_tokenizer.encode(text)
    print(integers)
    strings = orig_tokenizer.decode(integers)
    print(strings)
    
    
    print()