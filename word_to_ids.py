import glove, numpy as np

print('reading glove')
glove.get()
print('glove read')

def to_id(w):
    if w in glove.vocab:
        return glove.vocab[w]
    return None

def ws_to_ids(infile, outfile):
    with open(outfile, 'w') as with_ids, open(outfile + '_invalid_ws.txt', 'w') as invalid_ids:
        for line in open(infile, 'r'):
            if ',' not in line:
                with_ids.write(line)
            elif 'series:' in line:
                with_ids.write(line)
            else:
                ws = ' , '.join(line.split(',')[1:]).split()
                ids = []
                for w in ws:
                    id = to_id(w)
                    if id is None:
                        invalid_ids.write(w + ' is not found in the vocabulary\n')
                    else:
                        ids.append(str(id))
                with_ids.write(' '.join(ids) + '\n')


ws_to_ids('data/sp500_bloomberg_combined.txt', 'data/sp500_bloomberg_combined_with_ids.txt')
ws_to_ids('data/sp500_reuters_combined.txt', 'data/sp500_reuters_combined_with_ids.txt')
