import pandas as pd

TITLE_FILE_BLOOMBERG = '/Users/hjunlim/Documents/projects/newsfeed_openie/titles/20061020_20131126_bloomberg_news/titles_lemmatized.txt'
TITLE_FILE_REUTERS = '/Users/hjunlim/Documents/projects/newsfeed_openie/titles/ReutersNews106521/titles_lemmatized.txt'

sp_series = pd.read_csv('data/sp500_series.csv', index_col = 0)

with open('data/sp500_bloomberg_combined.txt', 'w') as combined_bloomberg:
    for line in open(TITLE_FILE_BLOOMBERG):
        if ',' not in line:
            combined_bloomberg.write(line)
            if line.strip() in sp_series.index:
                combined_bloomberg.write(
                    'series:' + ' '.join(map(lambda v: str(v), sp_series.loc[line.strip()].values)) + '\n')
        else:
            combined_bloomberg.write(line)

with open('data/sp500_reuters_combined.txt', 'w') as combined_reuters:
    for line in open(TITLE_FILE_REUTERS):
        if ',' not in line:
            combined_reuters.write(line)
            if line.strip() in sp_series.index:
                combined_reuters.write(
                    'series:' + ' '.join(map(lambda v: str(v), sp_series.loc[line.strip()].values)) + '\n')
        else:
            combined_reuters.write(line)
