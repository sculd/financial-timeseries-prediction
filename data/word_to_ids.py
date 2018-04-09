import glove as glove, numpy as np, re
from nltk.corpus import stopwords

print('reading glove')
glove.get()
print('glove read')

_STOP_WORDS = set(stopwords.words())

_DATE_REGEXP = re.compile('\d{4}-\d{2}-\d{2}')
def to_id(w):
    if w in glove.vocab:
        return glove.vocab[w]
    return None

def ws_to_ids(infile, outfile):
    with open(outfile, 'w') as with_ids, open(outfile.split('.txt')[0] + '_invalid_ws.txt', 'w') as invalid_ids:
        for line in open(infile, 'r'):
            if _DATE_REGEXP.match(line):
                with_ids.write(line)
            elif 'series:' in line:
                with_ids.write(line)
            else:
                ws = ' , '.join(line.split(',')[1:]).split()
                ids = []
                for w in ws:
                    if w in _STOP_WORDS:
                        continue
                    id = to_id(w)
                    if id is None:
                        invalid_ids.write(w + '\n')
                    else:
                        ids.append(str(id))
                with_ids.write(' '.join(ids) + '\n')

if __name__ == '__main__':
    ws_to_ids('data/sp500_bloomberg_combined.txt', 'data/sp500_bloomberg_combined_with_ids.txt')
    ws_to_ids('data/sp500_reuters_combined.txt', 'data/sp500_reuters_combined_with_ids.txt')
