"""Windows-native custom title-bar styling WITHOUT Tk override-redirect.

Keep a WM-managed window; modify only its visible nonclient caption style.
This preserves Alt+Tab, Taskbar and minimize/restore semantics.
"""
from __future__ import annotations
import sys

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_SYSMENU = 0x00080000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020


def stable_styles(style: int, exstyle: int) -> tuple[int,int]:
    """Pure version of Win32 style transformation (unit-testable on Linux)."""
    visible = (style & ~(WS_CAPTION | WS_THICKFRAME)) | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
    extended = (exstyle & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
    return visible, extended


def install_windows_chrome(window) -> bool:
    if sys.platform != 'win32':
        return False
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32',use_last_error=True)
    # Explicit 64-bit pointer types: the default ctypes return value is an int
    # and may truncate HWND on 64-bit Windows.
    get_parent = user32.GetParent
    get_parent.argtypes = (wintypes.HWND,)
    get_parent.restype = wintypes.HWND
    get_ancestor = user32.GetAncestor
    get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
    get_ancestor.restype = wintypes.HWND
    get_style = user32.GetWindowLongW
    get_style.argtypes = (wintypes.HWND, ctypes.c_int)
    get_style.restype = ctypes.c_long
    set_style = user32.SetWindowLongW
    set_style.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_long)
    set_style.restype = ctypes.c_long
    set_pos = user32.SetWindowPos
    set_pos.argtypes = (wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,
                        ctypes.c_int,ctypes.c_int,wintypes.UINT)
    set_pos.restype = wintypes.BOOL
    win_id=window.winfo_id()
    hwnd = get_ancestor(win_id,2) or get_parent(win_id) or win_id
    if not hwnd:
        return False
    style,exstyle=get_style(hwnd,GWL_STYLE),get_style(hwnd,GWL_EXSTYLE)
    desired,desired_ex=stable_styles(style,exstyle)
    if style!=desired:
        ctypes.set_last_error(0)
        prev=set_style(hwnd,GWL_STYLE,desired)
        if not prev and ctypes.get_last_error():
            raise OSError(ctypes.get_last_error(),'Failed to set Windows style')
    if exstyle!=desired_ex:
        ctypes.set_last_error(0)
        prev=set_style(hwnd,GWL_EXSTYLE,desired_ex)
        if not prev and ctypes.get_last_error():
            raise OSError(ctypes.get_last_error(),'Failed to set Windows taskbar style')
    if style!=desired or exstyle!=desired_ex:
        flags=SWP_NOMOVE|SWP_NOSIZE|SWP_NOZORDER|SWP_NOACTIVATE|SWP_FRAMECHANGED
        if not set_pos(hwnd,None,0,0,0,0,flags):
            raise OSError(ctypes.get_last_error(),'Failed to apply Windows frame changes')
    return True
