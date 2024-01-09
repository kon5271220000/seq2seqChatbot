from __future__ import absolute_import
from underthesea import word_tokenize
from collections import defaultdict
import numpy as np
import pickle

VI_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ '  # space is included in whitelist
VI_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'data.txt'

limit = {
    'maxq': 20,
    'minq': 0,
    'maxa': 20,
    'mina': 3
}

UNK = 'unk'
VOCAB_SIZE = 6000

def ddefault():
    return 1

def read_lines(filename):
    return open(filename, encoding='utf8').read().split('\n')[:-1]

def split_line(line):
    return line.split('.')

def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])

def index_(tokenized_sentences, vocab_size):
    # sử dụng hàm đếm tần suất từ của underthesea
    freq_dist = defaultdict(int)
    for sentence in tokenized_sentences:
        for word in sentence:
            freq_dist[word] += 1

    # get vocabulary của 'vocab_size' từ được sử dụng nhiều nhất
    vocab = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:vocab_size]

    # index2word
    index2word = ['_'] + [UNK] + [x[0] for x in vocab]
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dist

def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences) // 2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(word_tokenize(sequences[i])), len(word_tokenize(sequences[i + 1]))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i + 1])

    filt_data_len = len(filtered_q)
    print(str(len(filtered_q)))
    filtered = int((raw_data_len - filt_data_len) * 100 / raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

def zero_pad_sequence(sequence, max_length, padding_value=0):
    padded_sequence = sequence + [padding_value] * (max_length - len(sequence))
    return padded_sequence

def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = zero_pad_sequence([w2idx[word] if word in w2idx else w2idx[UNK] for word in qtokenized[i]], limit['maxq'])
        a_indices = zero_pad_sequence([w2idx[word] if word in w2idx else w2idx[UNK] for word in atokenized[i]], limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

def process_data():
    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)
    lines = [line.lower() for line in lines]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    print('\n>> Filter lines')
    lines = [filter_line(line, VI_WHITELIST) for line in lines]
    print(lines[121:125])

    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines)
    if qlines and alines:
        print('\nq : {0} ; a : {1}'.format(qlines[-1], alines[-1]))
    if qlines and alines:
        print('\nq : {0} ; a : {1}'.format(qlines[-1], alines[-1]))
    else:
        print("No data available.")



    print('\n>> Segment lines into words')
    qtokenized = [word_tokenize(wordlist) for wordlist in qlines]
    atokenized = [word_tokenize(wordlist) for wordlist in alines]
    print('\n:: Sample from segmented list of words')
    if qtokenized and atokenized:
        print('\nq : {0} ; a : {1}'.format(qtokenized[-1], atokenized[-1]))
    else:
        print("No data available.")
    if qtokenized and atokenized:
        print('\nq : {0} ; a : {1}'.format(qtokenized[-1], atokenized[-1]))
    else:
        print("No data available.")

    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

if __name__ == '__main__':
    process_data()