import time

class PerformanceTimer:
    def __init__(self) -> None:
        self.__total_time = 0
        
        self.__start_time = None
        
    def tic(self):
        self.__start_time = time.perf_counter()
        
    def toc(self):
        if self.__start_time is None:
            raise ValueError("Did you tic before tok?")
        self.__total_time += (time.perf_counter() - self.__start_time)
        self.__start_time = None
        
    def reset_total(self):
        self.__total_time = 0
        self.__start_time = None
    
    def get_total_time(self):
        return self.__total_time