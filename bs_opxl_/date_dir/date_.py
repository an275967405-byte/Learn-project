from datetime import datetime, timedelta

def get_date(days = 7 , format_str = '%Y-%m-%d'):
    data_days=[]
    day_ = range(days)
    for day_offset in day_:
        date = datetime.now() - timedelta(days=day_offset)
        data_days.append(date.strftime(format_str))
    return data_days

data_date_ = get_date(7)