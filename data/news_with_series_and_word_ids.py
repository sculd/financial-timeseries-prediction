import datetime, re, tensorflow as tf
_TIME_DELTA_ONE_DAY = datetime.timedelta(days = 1)
_DATE_REGEXP = re.compile('\d{4}-\d{2}-\d{2}')

FILENAME_BLOOMBERG = 'data/sp500_bloomberg_combined_with_ids.txt'
FILENAME_REUTERS = 'data/sp500_reuters_combined_with_ids.txt'

class Day:
    def __init__(self, date):
        self.date = date
        self.series = None
        self.ids_list = []
    def set_series(self, series):
        self.series = series
    def add_news(self, word_ids):
        self.ids_list.append(word_ids)
    def if_series_and_news(self):
        return self.series is not None and len(self.ids_list) > 0
    def pad_ids(self, maxlen = None):
        self.ids_list = tf.keras.preprocessing.sequence.pad_sequences(self.ids_list, maxlen = maxlen)
    def print(self):
        print(self.date)
        print(self.series)
        print('ids of length %d' % (len(self.ids_list)))

class Days:
    def __init__(self, filename):
        self.by_date = {} # datetime.date : Day
        date = None
        for line in open(filename, 'r'):
            if _DATE_REGEXP.match(line):
                y, m, d = map(lambda v: int(v), line.split('-'))
                date = datetime.date(year=y, month=m, day=d)
                self.by_date[date] = Day(date)
            elif 'series:' in line:
                series = list(map(lambda v: float(v), line.split('series:')[1].split()))
                self.by_date[date].series = series
            else:
                self.by_date[date].ids_list.append(list(map(lambda v: int(v), line.split())))

    def get_first_date(self):
        k_sorted = sorted(self.by_date.keys())
        return None if len(k_sorted) == 0 else k_sorted[0]

    def get_last_date(self):
        k_sorted = sorted(self.by_date.keys())
        return None if len(k_sorted) == 0 else k_sorted[-1]

    def if_date_in(self, date):
        return date in self.by_date

    def get_at(self, date):
        if not self.if_date_in(date):
            return None
        return self.by_date[date]

    def _get_max_ids_length(self):
        l = 0
        dit = DailyIter(self)
        while dit.has_next():
            day = dit.next()
            l = max(l, max([len(ids) for ids in day.ids_list]))
        return l

    def pad_ids(self):
        ml = self._get_max_ids_length()
        for _, d in self.by_date.items():
            d.pad_ids(ml)

    def save(self, filename):
        with open(filename, 'w') as outfile:
            k_sorted = sorted(self.by_date.keys())
            for date in k_sorted:
                if not self.if_date_in(date):
                    continue
                day = self.get_at(date)
                if not day.if_series_and_news():
                    continue
                outfile.write(str(date) + '\n')
                outfile.write('series:' +  ' '.join(map(lambda v:str(v), day.series)) + '\n')
                for ids in day.ids_list:
                    outfile.write(' '.join(map(lambda v: str(v), ids)) + '\n')

class DailyIter:
    def __init__(self, daily):
        self.daily = daily
        self.cur_date = daily.get_first_date()
        self.last_date = daily.get_last_date()
        self._seek()

    def _seek(self):
        while self.cur_date <= self.last_date:
            if not self.daily.if_date_in(self.cur_date):
                self.cur_date += _TIME_DELTA_ONE_DAY
            elif not self.daily.get_at(self.cur_date).if_series_and_news():
                self.cur_date += _TIME_DELTA_ONE_DAY
            else:
                break

    def has_next(self):
        if self.cur_date > self.last_date:
            return False
        return True

    def next(self):
        if not self.has_next():
            return None
        r = self.daily.get_at(self.cur_date)
        self.cur_date += _TIME_DELTA_ONE_DAY
        self._seek()
        return r

if __name__ == '__main__':
    dy = Days('sp500_reuters_combined_with_ids.txt')
    dit = DailyIter(dy)
    while dit.has_next():
        day = dit.next()
        day.print()
