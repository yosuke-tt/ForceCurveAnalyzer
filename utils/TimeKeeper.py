import sys
import time
from datetime import datetime
from datetime import timedelta


class TimeKeeper():
    def __init__(self, num_data: int):
        self.num_data: int = num_data
        self.start: float = time.time()
        self.num_called: int = 0

    def seconds2str(self, seconds: float) -> str:
        h: float = seconds / 3600
        m: float = (seconds - int(h) * 3600) / 60
        s: float = seconds - int(h) * 3600 - int(m) * 60
        strt: str
        if int(h) > 0:
            strt = "{:>4.1f}h".format(h)
        elif int(m) > 0:
            strt = "{:>4.1f}m".format(m)
        else:
            strt = "{:>4.1f}s".format(s)
        return strt

    def timeshow(self, logger=None) -> float:
        now: float = time.time()
        all_time: float = now - self.start
        loop_ave_time: float = all_time / (self.num_called + 1)

        rest_time: str = self.seconds2str(loop_ave_time * (self.num_data - self.num_called))
        pred_time = datetime.now() + timedelta(seconds=loop_ave_time * (self.num_data - self.num_called))

        time_keep = "{:>3}/{:>3} [{:<25}] ave : {:>3.1f} pred : {} rest : {}".format(
            (self.num_called + 1),
            self.num_data, "#" * (25 * (self.num_called + 1) // self.num_data),
            all_time / (self.num_called + 1),
            pred_time.strftime("%m/%d %H:%M:%S"),
            rest_time, end="")

        sys.stdout.write(time_keep + "\r")
        if (self.num_called + 1) != self.num_data:
            sys.stdout.flush()
        if logger is not None:
            logger.info(time_keep)

        self.num_called += 1
        return all_time


if __name__ == "__main__":
    tk = TimeKeeper(100)
    for i in range(100):
        tk.timeshow()
        time.sleep(0.1)
