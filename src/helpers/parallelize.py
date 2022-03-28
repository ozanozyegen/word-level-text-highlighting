from multiprocessing import Process, Queue

def run_in_separate_process(method, args):
    def queue_wrapper(q, params):
        r = method(*params)
        q.put(r)

    q = Queue()
    p = Process(target=queue_wrapper, args=(q, args))
    p.start()
    return_val = q.get()
    p.join()
    return return_val

