import os
import jieba
import re
from tqdm import tqdm

spam_files = os.listdir('./data/spam/')
stop_words = []
with open('./data/中文停用词表.txt', encoding='GBK') as fp:
    for line in fp.readlines():
        stop_words.append(line.replace('\n', ''))
#print(stop_words)

def get_word_set(file_name):
    rule = re.compile(r"[^\u4e00-\u9fa5]") #过滤掉非中文字符
    word_set = set()
    with open(file_name, encoding='GBK') as fp:
        for line in fp.readlines():
            line = rule.sub('', line)
            #print(line)
            words = list(jieba.cut(line))
            for word in words:
                if word != None and word not in stop_words and word.strip() != '':
                    word_set.add(word)
    return word_set

def stat_words(root_dir):
    '''
    统计邮件中出现各个词的次数
    比如某个词在垃圾邮件中出现了20次，在非垃圾邮件中出现了4次
    如果一个词在一封邮件出现多次，则只按一次计算
    这样就能计算P(词|垃圾)和P(词|非垃圾)了，分别记为pw_s和pw_n
    '''
    files = os.listdir(root_dir)
    word_cnt = {}
    for idx, f in enumerate(tqdm(files)):
        word_set = get_word_set(os.path.join(root_dir, f))
        for word in word_set:
            if word not in word_cnt.keys():
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
        #if idx > 200:
        #    break
    return len(files), word_cnt

n_normal_files, normal_word_cnt = stat_words('./data/normal/')
n_spam_files, spam_word_cnt = stat_words('./data/spam/')
#print(n_normal_files, normal_word_cnt)
#print(n_spam_files, spam_word_cnt)

correct, wrong = 0, 0
for f in os.listdir('./data/test/'):
    word_set = get_word_set(os.path.join('./data/test/', f))
    word_prob = {}
    for word in word_set:
        if word in spam_word_cnt.keys() and word in normal_word_cnt.keys():
            pw_s = spam_word_cnt[word] / n_spam_files
            pw_n = normal_word_cnt[word] / n_normal_files
        elif word in spam_word_cnt.keys():
            pw_s = spam_word_cnt[word] / n_spam_files
            pw_n = 0.01 #这个值是前人的经验值
        elif word in normal_word_cnt.keys():
            pw_s = 0.01
            pw_n = normal_word_cnt[word] / n_normal_files
        else:
            pw_s = 2
            pw_n = 3 #另ps_w=0.4，这个值也是前人的经验值
        ps_w = pw_s / (pw_s + pw_n)
        '''
        p(s|w) = p(w|s) * p(s) / p(w)
               = p(w|s) * p(s) / (p(w|s) * p(s) + p(w|n) * p(n))
               这里p(s)和p(n)相等，都是0.5，因为两种邮件数量基本一致，所以消去了
               = p(w|s) / (p(w|s) + p(w|n))
        '''
        word_prob[word] = ps_w
    sorted(word_prob.items(), key=lambda x : x[1], reverse=True)[0:15]
    #print(word_prob.items())
    ps_w_ = 1.0
    ps_n_ = 1.0
    for word, ps_w in word_prob.items():
        ps_w_ *= ps_w
        ps_n_ *= (1 - ps_w)
    final_ps_w = ps_w_ / (ps_w_ + ps_n_)

    if (int(f) > 1000 and final_ps_w > 0.9) or (int(f) < 1000 and final_ps_w <= 0.9):
        correct += 1
    else:
        wrong += 1
    print(int(f), final_ps_w)
print('correct:', correct, 'wrong:', wrong, 'accuracy:', correct / (correct + wrong))
