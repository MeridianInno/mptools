# -*- coding: utf-8 -*-

"""Main module."""

import functools
import logging
import multiprocessing as mp
import multiprocessing.queues as mpq
import signal
import sys
import time
from queue import Empty, Full
import numpy as np

DEFAULT_POLLING_TIMEOUT = 0.02
MAX_SLEEP_SECS = 0.02

start_time = time.monotonic()


def _logger(name, level, msg, exc_info=None):
    elapsed = time.monotonic() - start_time
    hours = int(elapsed // 60)
    seconds = elapsed - (hours * 60)
    _msg = '{:12s} {}'.format(name, msg)
    # logging.log(level, f'{hours:3}:{seconds:06.3f} {name:12} {msg}', exc_info=exc_info)
    logging.log(level, _msg, exc_info=exc_info)


# -- Queue handling support

class MPQueue(mpq.Queue):

    # -- See StackOverflow Article :
    #   https://stackoverflow.com/questions/39496554/cannot-subclass-multiprocessing-queue-in-python-3-5
    #
    # -- tldr; mp.Queue is a _method_ that returns an mpq.Queue object.  That object
    # requires a context for proper operation, so this __init__ does that work as well.
    def __init__(self, *args, **kwargs):
        ctx = mp.get_context()
        super().__init__(*args, **kwargs, ctx=ctx)

    def safe_get(self, timeout=DEFAULT_POLLING_TIMEOUT):
        try:
            if timeout is None:
                return self.get(block=False)
            else:
                return self.get(block=True, timeout=timeout)
        except Empty:
            return None

    def safe_put(self, item, timeout=DEFAULT_POLLING_TIMEOUT):
        try:
            self.put(item, block=False, timeout=timeout)
            return True
        except Full:
            return False

    def drain(self):
        item = self.safe_get()
        # note that 'while item' will cause problem if we have np.ndarray left
        # in the queue, for which the truth value cannot be defined.
        # But safe_get() returns None if queue is empty.
        # So the more explicit form below is preferred
        while item is not None:
            yield item
            item = self.safe_get()

    def safe_close(self):
        num_left = sum(1 for __ in self.drain())
        self.close()
        self.join_thread()
        return num_left


# -- useful function
def _sleep_secs(max_sleep, end_time=999999999999999.9):
    # Calculate time left to sleep, no less than 0
    return max(0.0, min(end_time - time.time(), max_sleep))


# -- Standard Event Queue manager
class EventMessage:
    def __init__(self, msg_src, msg_type, msg):
        self.id = time.time()
        self.msg_src = msg_src
        self.msg_type = msg_type
        self.msg = msg

    def __str__(self):
        return "{:10} - {:10} : {}".format(self.msg_src, self.msg_type, self.msg)


# -- Signal Handling
class TerminateInterrupt(BaseException):
    pass


class SignalObject:
    MAX_TERMINATE_CALLED = 3

    def __init__(self, shutdown_event):
        self.terminate_called = 0
        self.shutdown_event = shutdown_event


def default_signal_handler(signal_object, exception_class, signal_num, current_stack_frame):
    signal_object.terminate_called += 1
    signal_object.shutdown_event.set()
    if signal_object.terminate_called >= signal_object.MAX_TERMINATE_CALLED:
        raise exception_class()


def init_signal(signal_num, signal_object, exception_class, handler):
    handler = functools.partial(handler, signal_object, exception_class)
    signal.signal(signal_num, handler)
#    signal.siginterrupt(signal_num, False)


def init_signals(shutdown_event, int_handler, term_handler):
    signal_object = SignalObject(shutdown_event)
    init_signal(signal.SIGINT, signal_object, KeyboardInterrupt, int_handler)
    init_signal(signal.SIGTERM, signal_object, TerminateInterrupt, term_handler)
    return signal_object


# -- Worker Process classes

class ProcWorker:
    MAX_TERMINATE_CALLED = 3
    int_handler = staticmethod(default_signal_handler)
    term_handler = staticmethod(default_signal_handler)

    def __init__(self, name, startup_event, shutdown_event, event_q, *args):
        self.name = name
        self.log = functools.partial(_logger, '{}'.format(self.name))
        self.startup_event = startup_event
        self.shutdown_event = shutdown_event
        self.event_q = event_q
        self.terminate_called = 0
        self.init_args(args)

    def init_args(self, args):
        if args:
            raise ValueError("Unexpected arguments to ProcWorker.init_args: {}".format(args))

    def init_signals(self):
        self.log(logging.DEBUG, "{} Entering init_signals".format(self.name))
        signal_object = init_signals(self.shutdown_event, self.int_handler, self.term_handler)
        return signal_object

    def main_loop(self):
        # give some time for various other processes to start
        time.sleep(1)
        self.log(logging.DEBUG, "{} Entering main_loop".format(self.name))
        while not self.shutdown_event.is_set():
            self.main_func()
        self.log(logging.DEBUG, "Recieved shutdown event")

    def startup(self):
        self.log(logging.DEBUG, "{} Entering startup".format(self.name))
        pass

    def shutdown(self):
        self.log(logging.DEBUG, "{} Entering shutdown".format(self.name))
        pass

    def main_func(self, *args):
        self.log(logging.DEBUG, "Entering main_func")
        raise NotImplementedError("{}.main_func is not implemented".format(self.__class__.__name__))

    def run(self):
        self.init_signals()
        try:
            self.startup()
            self.startup_event.set()
            self.main_loop()
            self.log(logging.INFO, "Normal Shutdown")
            self.event_q.safe_put(EventMessage(self.name, "SHUTDOWN", "Normal"))
            return 0
        except BaseException as exc:
            # -- Catch ALL exceptions, even Terminate and Keyboard interrupt
            self.log(logging.ERROR, "Exception Shutdown: {}".format(exc), exc_info=True)
            self.event_q.safe_put(EventMessage(self.name, "FATAL", "{}".format(exc)))
            # -- TODO: call raise if in some sort of interactive mode
            if type(exc) in (TerminateInterrupt, KeyboardInterrupt):
                sys.exit(1)
            else:
                sys.exit(2)
        finally:
            self.shutdown()


class TimerProcWorker(ProcWorker):
    INTERVAL_SECS = 1  # 10
    MAX_SLEEP_SECS = 0.02

    def main_loop(self):
        self.log(logging.DEBUG, "Entering TimerProcWorker.main_loop")
        next_time = time.time() + self.INTERVAL_SECS
        while not self.shutdown_event.is_set():
            sleep_secs = _sleep_secs(self.MAX_SLEEP_SECS, next_time)
            time.sleep(sleep_secs)
            if time.time() > next_time:
                self.log(logging.DEBUG, "TimerProcWorker.main_loop : calling main_func")
                self.main_func()
                next_time = time.time() + self.INTERVAL_SECS


class QueueProcWorker(ProcWorker):
    def init_args(self, args):
        self.log(logging.DEBUG, "Entering QueueProcWorker.init_args : {}".format(args))
        self.work_q, = args
        print(self.work_q)

    def main_loop(self):
        self.log(logging.DEBUG, "Entering QueueProcWorker.main_loop")
        while not self.shutdown_event.is_set():
            item = self.work_q.safe_get()
            if not item:
                continue
            if not isinstance(item[1], np.ndarray):
                self.log(logging.DEBUG, "QueueProcWorker.main_loop received '{}' message".format(item))
            if item == "END":
                break
            else:
                self.main_func(item)


# -- Process Wrapper

def proc_worker_wrapper(proc_worker_class, name, startup_evt, shutdown_evt, event_q, *args):
    proc_worker = proc_worker_class(name, startup_evt, shutdown_evt, event_q, *args)
    return proc_worker.run()


class Proc:
    STARTUP_WAIT_SECS = 7.0
    SHUTDOWN_WAIT_SECS = 7.0

    def __init__(self, name, worker_class, shutdown_event, event_q, *args):
        self.log = functools.partial(_logger, '{}'.format(name))
        self.name = name
        self.shutdown_event = shutdown_event
        self.startup_event = mp.Event()
        self.proc = mp.Process(target=proc_worker_wrapper,
                               args=(worker_class, name, self.startup_event, shutdown_event, event_q, *args))
        self.log(logging.DEBUG, "Proc.__init__ starting : {}".format(name))
        self.proc.start()
        started = self.startup_event.wait(timeout=Proc.STARTUP_WAIT_SECS)
        self.log(logging.DEBUG, "Proc.__init__ starting : {} got {}".format(name, started))
        if not started:
            self.terminate()
            raise RuntimeError("Process {} failed to startup after {} seconds".
                               format(name, Proc.STARTUP_WAIT_SECS))

    def full_stop(self, wait_time=SHUTDOWN_WAIT_SECS):
        self.log(logging.DEBUG, "Proc.full_stop stoping : {}".format(self.name))
        self.shutdown_event.set()
        self.proc.join(wait_time)
        if self.proc.is_alive():
            self.terminate()

    def terminate(self):
        self.log(logging.DEBUG, "Proc.terminate terminating : {}".format(self.name))
        NUM_TRIES = 3
        tries = NUM_TRIES
        while tries and self.proc.is_alive():
            self.proc.terminate()
            time.sleep(0.01)
            tries -= 1

        if self.proc.is_alive():
            self.log(logging.ERROR, "Proc.terminate failed to terminate {} after {} attempts".
                    format(self.name, NUM_TRIES))
            return False
        else:
            self.log(logging.INFO, "Proc.terminate terminated {} after {} attempt(s)".
                    format(self.name, NUM_TRIES - tries))
            return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.full_stop()
        return not exc_type


# -- Main Wrappers
class MainContext:
    STOP_WAIT_SECS = 3.0

    def __init__(self):
        self.procs = []
        self.queues = []
        self.log = functools.partial(_logger, "MAIN")
        self.shutdown_event = mp.Event()
        self.event_queue = self.MPQueue()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.log(logging.ERROR, "Exception: {}".format(exc_val), exc_info=(exc_type, exc_val, exc_tb))

        self.log(logging.DEBUG, "MainContext.__eixt__: Stopping {} processes".format(len(self.procs)))
        self._stopped_procs_result = self.stop_procs()
        self._stopped_queues_result = self.stop_queues()

        # -- Don't eat exceptions that reach here.
        return not exc_type

    def Proc(self, name, worker_class, *args):
        proc = Proc(name, worker_class, self.shutdown_event, self.event_queue, *args)
        self.procs.append(proc)
        return proc

    def MPQueue(self, *args, **kwargs):
        q = MPQueue(*args, **kwargs)
        self.queues.append(q)
        return q

    def stop_procs(self):
        self.log(logging.DEBUG, 'MainContext.stop_procs: Stopping processes')
        self.shutdown_event.set()
        self.log(logging.DEBUG, '{} {}'.format(self.shutdown_event, self.shutdown_event.is_set()))
        end_time = time.time() + self.STOP_WAIT_SECS
        num_terminated = 0
        num_failed = 0

        # -- Wait up to STOP_WAIT_SECS for all processes to complete
        for proc in self.procs:
            join_secs = _sleep_secs(self.STOP_WAIT_SECS, end_time)
            proc.proc.join(join_secs)

        # -- Clear the procs list and _terminate_ any procs that
        # have not yet exited
        still_running = []
        while self.procs:
            proc = self.procs.pop()
            if proc.proc.is_alive():
                if proc.terminate():
                    num_terminated += 1
                else:
                    still_running.append(proc)
            else:
                exitcode = proc.proc.exitcode
                if exitcode:
                    self.log(logging.ERROR, "Process {} ended with exitcode {}".
                             format(proc.name, exitcode))
                    num_failed += 1
                else:
                    self.log(logging.DEBUG, "Process {} stopped successfully".
                            format(proc.name))

        self.procs = still_running
        return num_failed, num_terminated

    def stop_queues(self):
        num_items_left = 0
        # -- Clear the queues list and close all associated queues
        for q in self.queues:
            num_items_left += sum(1 for __ in q.drain())
            q.close()

        # -- Wait for all queue threads to stop
        while self.queues:
            q = self.queues.pop(0)
            q.join_thread()
        return num_items_left
