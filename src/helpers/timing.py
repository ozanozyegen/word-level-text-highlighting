import datetime

def get_hr_min_sec_str():
    now = datetime.datetime.now()
    time_str = f"{now.hour}:{now.minute}:{now.second}"
    return time_str
