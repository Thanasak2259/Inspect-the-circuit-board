"""Launch with persistent error logs for Windows GUI troubleshooting."""
from pcb_inspector.diagnostics import setup_diagnostics, install_tk_exception_logging
logger=setup_diagnostics()
try:
    from pcb_inspector.gui_simple import App
    app=App()
    install_tk_exception_logging(app)
    app.mainloop()
    if getattr(app,'_closing',False):
        logger.info('Application exited after Close was pressed')
    else:
        logger.error('GUI event loop exited unexpectedly without Close action')
except BaseException:
    logger.exception('Application terminated unexpectedly')
    raise
