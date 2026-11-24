"""Crash diagnostics kept alongside the app for Windows user bug reports."""
from __future__ import annotations
import faulthandler
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
import traceback

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
_FAULT_STREAM = None


def setup_diagnostics():
    global _FAULT_STREAM
    LOG_DIR.mkdir(exist_ok=True,parents=True)
    logger=logging.getLogger('pcb_inspection')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler=RotatingFileHandler(LOG_DIR/'pcb_inspection.log',maxBytes=1_000_000,backupCount=2,encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(message)s'))
        logger.addHandler(handler)
    if _FAULT_STREAM is None:
        _FAULT_STREAM=open(LOG_DIR/'fatal_traceback.log','a',encoding='utf-8')
        try:faulthandler.enable(file=_FAULT_STREAM,all_threads=True)
        except Exception: logger.exception('Cannot enable Python faulthandler')
    original=sys.excepthook
    def on_exception(kind,value,tb):
        logger.error('Uncaught error',exc_info=(kind,value,tb))
        original(kind,value,tb)
    sys.excepthook=on_exception
    if hasattr(threading,'excepthook'):
        original_thread=threading.excepthook
        def on_thread_error(args):
            logger.error('Uncaught background error',exc_info=(args.exc_type,args.exc_value,args.exc_traceback))
            original_thread(args)
        threading.excepthook=on_thread_error
    logger.info('Application start (Python %s, platform %s)',sys.version.split()[0],sys.platform)
    return logger


def install_tk_exception_logging(window):
    logger=logging.getLogger('pcb_inspection')
    def callback(kind,value,tb):
        logger.error('Tkinter callback failed',exc_info=(kind,value,tb))
        traceback.print_exception(kind,value,tb)
        try:
            from tkinter import messagebox
            messagebox.showerror('PCB Inspection error',
                'An unexpected error occurred. A report was saved in logs/pcb_inspection.log.')
        except Exception:
            pass
    window.report_callback_exception=callback
