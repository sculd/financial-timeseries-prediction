import re
from nltk.corpus import stopwords

_STOP_WORDS = set(stopwords.words('english')).union({',', ';', ':', "'s"})
_DATE_REGEXP = re.compile('\d{4}-\d{2}-\d{2}')

def titles(infile, outfile):
    with open(outfile, 'w') as with_embedding:
        for line in open(infile, 'r'):
            if _DATE_REGEXP.match(line):
                #with_embedding.write(line)
                pass
            elif 'series:' in line:
                #with_embedding.write(line)
                pass
            else:
                ws = ' , '.join(line.split(',')[1:]).split()
                ws = [w for w in ws if w not in _STOP_WORDS]
                sentence = ' '.join(ws)
                #with_embedding.write(line.split(',')[0] + ',')
                with_embedding.write(sentence + '\n')

titles('sp500_bloomberg_combined.txt', 'sp500_bloomberg_combined_.txt')
titles('sp500_reuters_combined.txt', 'sp500_reuters_combined_.txt')
