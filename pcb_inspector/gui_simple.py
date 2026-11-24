"""One-click PCB inspection with pixel-night / starfield desktop theme."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import queue
import logging
import sys
from datetime import datetime
import cv2
from PIL import Image, ImageTk
from .core import inspect, load_image, save_report
from .comparison_viewer import ComparisonViewer
from .finding_overlay import render_findings

BG='#130724'; PANEL='#201238'; PANEL2='#2d1b49'; STROKE='#574176'; MUTED='#b7a5d1'
TEXT='#f9f2ff'; BLUE='#a878ff'; RED='#ff7fbc'; AMBER='#ffce8b'; GREEN='#9ce7ca'
FONT='Segoe UI'

class Preview(tk.Frame):
    def __init__(self, parent, title, subtitle='', **kw):
        super().__init__(parent,bg=PANEL,highlightbackground=STROKE,highlightthickness=1,**kw)
        head=tk.Frame(self,bg=PANEL);head.pack(fill='x',padx=14,pady=(12,5))
        tk.Label(head,text=title,bg=PANEL,fg=TEXT,font=(FONT,11,'bold')).pack(anchor='w')
        if subtitle:tk.Label(head,text=subtitle,bg=PANEL,fg=MUTED,font=(FONT,9)).pack(anchor='w',pady=(2,0))
        self.canvas=tk.Canvas(self,bg=PANEL,bd=0,highlightthickness=0)
        self.canvas.pack(fill='both',expand=True,padx=10,pady=(4,10))
        self.raw=None;self.photo=None
        self.zoom=1.0;self.pan_x=0;self.pan_y=0;self.drag_anchor=None
        self.canvas.bind('<Configure>',lambda event:self.render())
        self.canvas.bind('<MouseWheel>',self._on_wheel)
        self.canvas.bind('<Button-4>',lambda event:self._zoom(1.15))
        self.canvas.bind('<Button-5>',lambda event:self._zoom(1/1.15))
        self.canvas.bind('<ButtonPress-1>',self._drag_start)
        self.canvas.bind('<B1-Motion>',self._drag_move)
        self.canvas.bind('<Double-Button-1>',self._fit)

    def set(self,image):
        self.raw=image;self.zoom=1.0;self.pan_x=self.pan_y=0;self.render()

    def _zoom(self,multiplier):
        if self.raw is None:return
        self.zoom=max(1.0,min(10.0,self.zoom*multiplier));self.render()

    def _on_wheel(self,event):
        self._zoom(1.15 if event.delta>0 else 1/1.15)

    def _drag_start(self,event):
        self.drag_anchor=(event.x,event.y,self.pan_x,self.pan_y)

    def _drag_move(self,event):
        if self.raw is None or self.drag_anchor is None:return
        ax,ay,px,py=self.drag_anchor
        self.pan_x=px+event.x-ax;self.pan_y=py+event.y-ay;self.render()

    def _fit(self,event=None):
        self.zoom=1.0;self.pan_x=self.pan_y=0;self.render()

    def render(self):
        c=self.canvas;c.delete('all');w=c.winfo_width();h=c.winfo_height()
        if w<30 or h<30:return
        if self.raw is None:
            # Tiny, deterministic pixel stars only in EMPTY preview areas.
            for fx,fy,r in ((.12,.18,1),(.25,.69,1),(.71,.16,2),(.86,.76,1),(.53,.32,1),(.36,.15,1),(.9,.29,1)):
                x=int(w*fx);y=int(h*fy)
                c.create_rectangle(x-r,y-r,x+r,y+r,fill='#c9b3f9',outline='')
            c.create_text(w/2,h/2-12,text='✧',fill='#9d83ce',font=(FONT,30))
            c.create_text(w/2,h/2+24,text='No image selected',fill=MUTED,font=(FONT,10))
            return
        arr=cv2.cvtColor(self.raw,cv2.COLOR_BGR2RGB)
        im=Image.fromarray(arr)
        fit=min(max(w-12,1)/im.width,max(h-12,1)/im.height)
        scale=fit*self.zoom
        rw=max(1,int(im.width*scale));rh=max(1,int(im.height*scale))
        # Limit extreme zoom raster sizes; canvas clips overflow to its viewport.
        if rw*rh > 24_000_000:
            factor=(24_000_000/(rw*rh))**0.5
            rw=max(1,int(rw*factor));rh=max(1,int(rh*factor))
        im=im.resize((rw,rh),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(im)
        c.create_image(w/2+self.pan_x,h/2+self.pan_y,image=self.photo,anchor='center')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PCB Inspection  ·  Moonlit Pixel Studio  v0.9.22')
        self.geometry('1380x880');self.minsize(1000,690);self.configure(bg=BG)
        # Windows must remain managed by its window manager for stable Alt+Tab,
        # minimization, taskbar and focus restoration. Remove only visual chrome
        # with Win32 styles; NEVER turn override-redirect back on for Windows.
        self._native_window_manager = sys.platform == 'win32'
        if not self._native_window_manager:
            self.overrideredirect(True)
        self._maximized=False
        self._restore_geometry=None
        self._drag_start=None
        self._resize_start=None
        self._minimizing=False
        self._chrome_installed=False
        self._work_queue=queue.SimpleQueue()
        self._closing=False
        self.bind('<Map>',self._on_map,add='+')
        self.bind('<FocusOut>',self._focus_out,add='+')
        self.bind('<FocusIn>',self._focus_in,add='+')
        self.bind('<Escape>',lambda e:self._restore() if self._maximized else None)
        self.protocol('WM_DELETE_WINDOW',self._close)
        if self._native_window_manager:
            self.after(120,self._install_windows_chrome)
        else:
            self.after(150,self._show_in_taskbar)
        self.after(100,self._poll_worker_events)
        self.ref_path='';self.test_path='';self.result=None
        self._dismiss_history=[];self._finding_widgets={}
        self.preview_images={};self.busy=False;self._job_id=0
        self._sky_source=Image.open(Path(__file__).resolve().parent.parent/'assets'/'moonlit_header.png').convert('RGB')
        self._sky_photo=None
        self._build()

    def label(self,parent,text,size=10,fg=TEXT,bold=False,bg=None,**opts):
        return tk.Label(parent,text=text,font=(FONT,size,'bold' if bold else 'normal'),fg=fg,bg=bg or parent.cget('bg'),**opts)

    def button(self,parent,text,command,primary=False):
        b=tk.Button(parent,text=text,command=command,cursor='hand2',font=(FONT,10,'bold'),
                    bg=BLUE if primary else PANEL2,fg='#17092c' if primary else TEXT,
                    activebackground='#ccb1ff' if primary else '#473066',activeforeground='#17092c' if primary else TEXT,
                    relief='flat',bd=0,padx=19,pady=11,highlightthickness=0)
        return b

    def _build(self):
        self._build_titlebar()
        # User-supplied pixel-night artwork: moon, clouds and real stars.
        # The header rerenders on resize; buttons remain normal, accessible Tk controls.
        self.sky=tk.Canvas(self,height=190,bg=BG,highlightthickness=0,bd=0)
        self.sky.pack(fill='x',padx=0,pady=(0,0))
        self.sky.bind('<Configure>',self._render_sky)
        self.export_btn=self.button(self.sky,'Export report',self.export)
        self.export_btn.config(state='disabled')
        self.inspect_btn=self.button(self.sky,'✦  Inspect PCB',self.run,primary=True)
        self._sky_inspect_window=self.sky.create_window(0,0,window=self.inspect_btn,anchor='se')
        self._sky_export_window=self.sky.create_window(0,0,window=self.export_btn,anchor='se')

        inputs=tk.Frame(self,bg=BG);inputs.pack(fill='x',padx=26,pady=(0,14))
        inputs.grid_columnconfigure((0,1),weight=1,uniform='a')
        for i,(title,sub,which) in enumerate((('01  REFERENCE','Completed PCB / master image','ref'),('02  INSPECTION','PCB that needs inspection','test'))):
            card=tk.Frame(inputs,bg=PANEL,highlightbackground=STROKE,highlightthickness=1)
            card.grid(row=0,column=i,sticky='ew',padx=(0,8) if i==0 else (8,0))
            inner=tk.Frame(card,bg=PANEL);inner.pack(fill='x',padx=17,pady=12)
            line=tk.Frame(inner,bg=PANEL);line.pack(fill='x')
            self.label(line,title,10,bold=True).pack(side='left')
            self.button(line,'Choose image',lambda k=which:self.browse(k)).pack(side='right')
            v=self.label(inner,'No file selected',9,fg=MUTED,anchor='w')
            v.pack(fill='x',pady=(6,0));self.preview_images[which+'_filename']=v
        tools=tk.Frame(self,bg=BG);tools.pack(fill='x',padx=27,pady=(0,6))
        self.swap_btn=self.button(tools,'⇄  Swap images',self.swap_images)
        self.swap_btn.pack(side='left')
        self.reset_btn=self.button(tools,'Clear',self.clear_images)
        self.reset_btn.pack(side='left',padx=(8,0))
        self.undo_btn=self.button(tools,'Undo dismissed',self.undo_dismissed)
        self.undo_btn.config(state='disabled')
        self.undo_btn.pack(side='left',padx=(8,0))
        self.label(tools,'Board: wheel zoom / drag pan / double-click fit. Click a finding for comparison.',9,fg=MUTED).pack(side='right')
        body=tk.PanedWindow(self,orient='horizontal',bg=BG,bd=0,sashwidth=8,showhandle=False)
        body.pack(fill='both',expand=True,padx=22,pady=(0,10))
        image_area=tk.Frame(body,bg=BG);body.add(image_area,minsize=560,stretch='always')
        sidebar=tk.Frame(body,bg=PANEL,highlightbackground=STROKE,highlightthickness=1);body.add(sidebar,minsize=295,width=340)
        image_area.grid_columnconfigure((0,1),weight=1,uniform='img')
        image_area.grid_rowconfigure(0,weight=3);image_area.grid_rowconfigure(1,weight=2)
        self.ref_preview=Preview(image_area,'Reference board','Original master image')
        self.test_preview=Preview(image_area,'Inspection board','Original image before alignment')
        self.result_preview=Preview(image_area,'Detection results','Red = likely missing  ·  Amber = review  ·  Violet = masked')
        self.ref_preview.grid(row=0,column=0,sticky='nsew',padx=4,pady=4)
        self.test_preview.grid(row=0,column=1,sticky='nsew',padx=4,pady=4)
        self.result_preview.grid(row=1,column=0,columnspan=2,sticky='nsew',padx=4,pady=4)
        self.compare_viewer=ComparisonViewer(image_area,on_back=self.close_comparison,on_dismiss=self.dismiss_finding)
        self.compare_viewer.grid(row=1,column=0,columnspan=2,sticky='nsew',padx=4,pady=4)
        self.compare_viewer.grid_remove()
        self.label(sidebar,'✦  INSPECTION SUMMARY',11,bold=True).pack(anchor='w',padx=18,pady=(19,10))
        metrics=tk.Frame(sidebar,bg=PANEL);metrics.pack(fill='x',padx=14,pady=(0,10))
        for i,(name,color) in enumerate((('LIKELY MISSING',RED),('NEEDS REVIEW',AMBER))):
            box=tk.Frame(metrics,bg=PANEL2)
            box.grid(row=0,column=i,sticky='nsew',padx=4)
            metrics.grid_columnconfigure(i,weight=1)
            value=self.label(box,'—',26,fg=color,bold=True,bg=PANEL2)
            value.pack(anchor='w',padx=12,pady=(9,0))
            self.label(box,name,8,fg=MUTED,bg=PANEL2).pack(anchor='w',padx=12,pady=(0,10))
            self.preview_images['metric_'+str(i)]=value
        self.label(sidebar,'✧  FINDINGS',10,bold=True).pack(anchor='w',padx=18,pady=(10,8))
        listwrap=tk.Frame(sidebar,bg=PANEL);listwrap.pack(fill='both',expand=True,padx=14)
        self.list_canvas=tk.Canvas(listwrap,bg=PANEL,highlightthickness=0)
        # Custom-drawn scrollbar, not the bright native Windows scrollbar.
        scroll=tk.Canvas(listwrap,width=10,bg=PANEL,highlightthickness=0,bd=0,cursor='hand2')
        self._scroll_track=scroll
        self._scroll_thumb=None
        scroll.bind('<Button-1>',self._scroll_click)
        scroll.bind('<B1-Motion>',self._scroll_drag)
        scroll.bind('<Configure>',lambda e:self._update_scrollbar())
        self.items=tk.Frame(self.list_canvas,bg=PANEL)
        self.items.bind('<Configure>',lambda e:self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all')))
        self.window_id=self.list_canvas.create_window((0,0),window=self.items,anchor='nw')
        self.list_canvas.bind('<Configure>',lambda e:self.list_canvas.itemconfigure(self.window_id,width=e.width))
        self.list_canvas.configure(yscrollcommand=self._update_scrollbar)
        self.list_canvas.bind('<MouseWheel>',self._wheel)
        self.list_canvas.bind('<Button-4>',lambda e:self._wheel_linux(-1))
        self.list_canvas.bind('<Button-5>',lambda e:self._wheel_linux(1))
        self.items.bind('<MouseWheel>',self._wheel)
        self.items.bind('<Button-4>',lambda e:self._wheel_linux(-1))
        self.items.bind('<Button-5>',lambda e:self._wheel_linux(1))
        self.list_canvas.pack(side='left',fill='both',expand=True);scroll.pack(side='right',fill='y')
        self.label(self.items,'Results will appear here after inspection.',9,fg=MUTED,wraplength=280,justify='left').pack(anchor='w',padx=7,pady=12)
        self.label(sidebar,'Red = part gone (bare board where a part used to be). Yellow = changed area, please check. Wrong box? Click it, then "Not a defect".',9,fg=MUTED,wraplength=290,justify='left').pack(anchor='w',padx=18,pady=14)
        footer=tk.Frame(self,bg=BG);footer.pack(fill='x',padx=26,pady=(0,14))
        self.status=self.label(footer,'●  Ready — choose 2 images to begin.',9,fg=MUTED,anchor='w')
        self.status.pack(side='left',fill='x',expand=True)
        grip=self.label(footer,'◢',13,fg='#735796',anchor='se')
        grip.pack(side='right',padx=(12,0))
        grip.configure(cursor='size_nw_se' if __import__('sys').platform=='win32' else 'bottom_right_corner')
        grip.bind('<ButtonPress-1>',self._start_resize)
        grip.bind('<B1-Motion>',self._do_resize)
        self.label(footer,'✦  AUTO ALIGN  /  AUTO DETECT  /  NO MANUAL SETTINGS  ✦',9,fg='#ac91d7').pack(side='right')

    def _build_titlebar(self):
        # Made of Tk widgets, so it works with Python 3.14 / Tk on Windows.
        bar=tk.Frame(self,bg='#160a2c',height=39)
        bar.pack(fill='x',side='top')
        bar.pack_propagate(False)
        left=tk.Frame(bar,bg='#160a2c')
        left.pack(side='left',fill='y',padx=(15,0))
        icon=tk.Label(left,text='✦',font=(FONT,15,'bold'),fg='#b894ff',bg='#160a2c')
        icon.pack(side='left',padx=(0,10))
        title=tk.Label(left,text='PCB Inspection  ·  Moonlit Pixel Studio',font=(FONT,10,'bold'),fg=TEXT,bg='#160a2c')
        title.pack(side='left')
        self._max_btn=None
        for glyph,action in [('—',self._minimize),('□',self._toggle_maximize),('×',self._close)]:
            is_close=glyph=='×'
            b=tk.Label(bar,text=glyph,font=(FONT,14),fg='#dfcdfb',bg='#160a2c',
                       width=5,cursor='hand2')
            b.pack(side='right',fill='y')
            b.bind('<Button-1>',lambda e,fn=action:fn())
            b.bind('<Enter>',lambda e,w=b,c=is_close:w.configure(bg='#a82f68' if c else '#35204f'))
            b.bind('<Leave>',lambda e,w=b:w.configure(bg='#160a2c'))
            if glyph=='□':self._max_btn=b
        for w in (bar,left,icon,title):
            w.bind('<ButtonPress-1>',self._start_drag)
            w.bind('<B1-Motion>',self._do_drag)
            w.bind('<Double-Button-1>',lambda e:self._toggle_maximize())

    def _start_drag(self,event):
        if self._maximized:return
        self._drag_start=(event.x_root-self.winfo_x(),event.y_root-self.winfo_y())

    def _do_drag(self,event):
        if self._drag_start is None or self._maximized:return
        ox,oy=self._drag_start
        self.geometry(f'+{event.x_root-ox}+{event.y_root-oy}')

    def _work_area(self):
        if __import__('sys').platform=='win32':
            try:
                import ctypes
                from ctypes import wintypes
                rect=wintypes.RECT()
                if ctypes.windll.user32.SystemParametersInfoW(48,0,ctypes.byref(rect),0):
                    return rect.left,rect.top,rect.right-rect.left,rect.bottom-rect.top
            except Exception:pass
        return 0,0,self.winfo_screenwidth(),self.winfo_screenheight()

    def _toggle_maximize(self):
        if self._maximized:self._restore()
        else:
            self._restore_geometry=self.geometry()
            x,y,w,h=self._work_area()
            self.geometry(f'{w}x{h}+{x}+{y}')
            self._maximized=True
            if self._max_btn:self._max_btn.configure(text='❐')

    def _restore(self):
        if not self._maximized:return
        self._maximized=False
        if self._restore_geometry:self.geometry(self._restore_geometry)
        if self._max_btn:self._max_btn.configure(text='□')

    def _close(self):
        self._closing=True
        logging.getLogger('pcb_inspection').info('Window closed by user')
        self.destroy()

    def _focus_out(self,event):
        if event.widget is self:
            logging.getLogger('pcb_inspection').info('Window focus lost')

    def _focus_in(self,event):
        if event.widget is self:
            logging.getLogger('pcb_inspection').info('Window focus restored')

    def _minimize(self):
        # A window managed by Windows can safely iconify without flipping
        # override-redirect while Alt+Tab is being processed.
        self._minimizing=True
        if not self._native_window_manager:
            self.overrideredirect(False)
        self.iconify()
        logging.getLogger('pcb_inspection').info('Window minimized')

    def _on_map(self,event):
        if event.widget is not self or self._closing:
            return
        if self._minimizing:
            self._minimizing=False
            if not self._native_window_manager:
                self.after(20,lambda:self.overrideredirect(True))
            else:
                self.after(100,self._install_windows_chrome)
            logging.getLogger('pcb_inspection').info('Window mapped after minimize')

    def _install_windows_chrome(self):
        if self._closing or not self.winfo_exists() or not self._native_window_manager:
            return
        try:
            from .window_chrome import install_windows_chrome
            self._chrome_installed=install_windows_chrome(self)
            logging.getLogger('pcb_inspection').info(
                'Windows chrome configured: %s',self._chrome_installed)
        except Exception:
            # Keep the native title bar if Win32 styling fails rather than
            # risking an invisible or unusable application window.
            self._chrome_installed=False
            logging.getLogger('pcb_inspection').exception('Windows custom chrome failed; native frame retained')

    def _show_in_taskbar(self):
        if self._native_window_manager:
            self._install_windows_chrome()

    def _poll_worker_events(self):
        if self._closing:return
        try:
            while True:
                kind,job_id,payload=self._work_queue.get_nowait()
                if job_id != self._job_id:
                    continue
                if kind=='progress':
                    self.status.configure(text='●  '+payload)
                elif kind=='failed':
                    self.failed(payload,job_id)
                elif kind=='finished':
                    self.finished(payload,job_id)
        except queue.Empty:
            pass
        except Exception:
            logging.getLogger('pcb_inspection').exception('GUI event processing failed')
        finally:
            if not self._closing:
                self.after(100,self._poll_worker_events)

    def _start_resize(self,event):
        if self._maximized:return
        self._resize_start=(event.x_root,event.y_root,self.winfo_width(),self.winfo_height())

    def _do_resize(self,event):
        if not self._resize_start or self._maximized:return
        sx,sy,w,h=self._resize_start
        self.geometry(f'{max(1000,w+event.x_root-sx)}x{max(690,h+event.y_root-sy)}')

    def _update_scrollbar(self,first=None,last=None):
        if not hasattr(self,'_scroll_track'):return
        track=self._scroll_track
        track.delete('all')
        h=max(1,track.winfo_height())
        if first is None or last is None:
            first,last=self.list_canvas.yview()
        else:first,last=float(first),float(last)
        if last-first>=0.999:return
        y1=int(h*first);y2=max(y1+20,int(h*last))
        track.create_rectangle(3,y1,8,min(h,y2),fill='#76549c',outline='',tags='thumb')

    def _scroll_click(self,event):
        h=max(1,self._scroll_track.winfo_height())
        self.list_canvas.yview_moveto(max(0,min(1,event.y/h)))

    def _scroll_drag(self,event):
        h=max(1,self._scroll_track.winfo_height())
        first,last=self.list_canvas.yview()
        visible=last-first
        self.list_canvas.yview_moveto(max(0,min(1-visible,event.y/h-visible/2)))

    def _wheel(self,event):
        self.list_canvas.yview_scroll(-int(event.delta/120) if event.delta else 0,'units')
        return 'break'

    def _wheel_linux(self,direction):
        self.list_canvas.yview_scroll(direction,'units')
        return 'break'

    def _render_sky(self,event=None):
        width=max(1,self.sky.winfo_width())
        height=max(1,self.sky.winfo_height())
        if width<50:return
        # Preserving the whole aspect ratio of the moon is important: the
        # original v0.9.1 header stretched a square image to a panorama.
        # This precomposed landscape asset uses an unstretched moon.
        art=self._sky_source.resize((width,height),Image.Resampling.BILINEAR)
        from PIL import ImageDraw
        overlay=Image.new('RGBA',(width,height),(0,0,0,0))
        draw=ImageDraw.Draw(overlay)
        for x in range(min(width,780)):
            opacity=int(100*max(0,1-x/780))
            draw.line([(x,0),(x,height)],fill=(12,2,30,opacity))
        art=Image.alpha_composite(art.convert('RGBA'),overlay)
        self._sky_photo=ImageTk.PhotoImage(art)
        self.sky.delete('sky-art')
        self.sky.create_image(0,0,image=self._sky_photo,anchor='nw',tags=('sky-art',))
        self.sky.create_text(38,51,text='✦  PCB INSPECTION',fill='#ffffff',
                             font=(FONT,23,'bold'),anchor='w',tags=('sky-art',))
        self.sky.create_text(41,91,text='MOONLIT STUDIO  /  AUTOMATED IMAGE PROCESSING',
                             fill='#e2cbff',font=(FONT,10,'bold'),anchor='w',tags=('sky-art',))
        self.sky.create_text(41,129,text='✧  Reference  →  Inspection  →  Results',
                             fill='#dfc2fa',font=(FONT,10),anchor='w',tags=('sky-art',))
        self.sky.tag_lower('sky-art')
        self.sky.coords(self._sky_inspect_window,width-23,height-22)
        self.sky.coords(self._sky_export_window,width-184,height-22)

    def _reset_findings(self):
        self._dismiss_history.clear()
        self._finding_widgets.clear()
        self.undo_btn.config(state='disabled')
        self.close_comparison()
        self.result=None
        self.export_btn.configure(state='disabled')
        self.result_preview.set(None)
        for i in range(2):
            self.preview_images['metric_'+str(i)].configure(text='—')
        for child in self.items.winfo_children():
            child.destroy()
        self.label(self.items,'Results will appear here after inspection.',9,fg=MUTED,wraplength=280).pack(anchor='w',padx=7,pady=12)
        self.list_canvas.yview_moveto(0)
        self._update_scrollbar()

    def browse(self,which):
        if self.busy:return
        path=filedialog.askopenfilename(title='Select PCB image',filetypes=[('Images','*.png *.jpg *.jpeg *.bmp *.tif *.tiff'),('All files','*.*')])
        if not path:return
        try:im=load_image(path)
        except Exception as exc:messagebox.showerror('Cannot open image',str(exc));return
        if which=='ref':self.ref_path=path;self.ref_preview.set(im)
        else:self.test_path=path;self.test_preview.set(im)
        self.preview_images[which+'_filename'].config(text=Path(path).name)
        self._job_id+=1
        self._reset_findings()
        self.status.config(text='●  Images ready — click Inspect PCB.' if self.ref_path and self.test_path else '●  Choose the second image.')

    def swap_images(self):
        if self.busy:return
        self.ref_path,self.test_path=self.test_path,self.ref_path
        self.ref_preview.raw,self.test_preview.raw=self.test_preview.raw,self.ref_preview.raw
        self.ref_preview.render();self.test_preview.render()
        for which,path in (('ref',self.ref_path),('test',self.test_path)):
            self.preview_images[which+'_filename'].configure(text=Path(path).name if path else 'No file selected')
        self._job_id+=1;self._reset_findings()
        if self.ref_path:self.ref_preview.set(load_image(self.ref_path))
        if self.test_path:self.test_preview.set(load_image(self.test_path))
        self.status.configure(text='●  Images swapped. Run inspection again.')

    def clear_images(self):
        if self.busy:return
        self.ref_path=self.test_path=''
        self.ref_preview.set(None);self.test_preview.set(None)
        for which in ('ref','test'):
            self.preview_images[which+'_filename'].configure(text='No file selected')
        self._job_id+=1;self._reset_findings()
        self.status.configure(text='●  Ready — choose 2 images to begin.')

    def run(self):
        if not self.ref_path or not self.test_path:
            messagebox.showinfo('Select two images','Choose the reference PCB and the inspection PCB first.');return
        if self.busy:return
        if self.ref_path==self.test_path:
            messagebox.showwarning('Same file selected','Choose two different images to inspect.');return
        self.busy=True;self._job_id+=1;job_id=self._job_id
        self._reset_findings()
        self.inspect_btn.config(state='disabled',text='Inspecting...')
        self.swap_btn.config(state='disabled');self.reset_btn.config(state='disabled')
        self.status.config(text='●  Aligning boards and comparing component regions...')
        p1,p2=self.ref_path,self.test_path
        def worker():
            try:
                reference=load_image(p1)
                inspected=load_image(p2)
                def model_progress(text):
                    # Tk is not thread-safe. The worker must never call .after(),
                    # configure(), or any other Tk method, even when unfocused.
                    self._work_queue.put(('progress',job_id,str(text)))
                result=inspect(reference,inspected)
            except Exception as exc:
                logging.getLogger('pcb_inspection').exception('Inspection worker failed')
                self._work_queue.put(('failed',job_id,str(exc)))
                return
            self._work_queue.put(('finished',job_id,result))
        threading.Thread(target=worker,daemon=True).start()

    def _end_busy(self):
        self.busy=False
        self.inspect_btn.config(state='normal',text='✦  Inspect PCB')
        self.swap_btn.config(state='normal');self.reset_btn.config(state='normal')

    def failed(self,msg,job_id):
        if job_id!=self._job_id:return
        self._end_busy()
        self.status.config(text='●  Unable to inspect: '+msg)
        messagebox.showerror('Inspection error',msg)

    def finished(self,result,job_id):
        if job_id!=self._job_id:return
        self._end_busy()
        self.close_comparison()
        self.result=result;self.result_preview.set(result.overlay)
        self.export_btn.config(state='normal')
        findings=[(i,f) for i,f in enumerate(result.findings,1) if f.review_status!='Low-priority change']
        likely=sum(f.review_status=='Likely missing' for _,f in findings)
        self.preview_images['metric_0'].config(text=str(likely))
        self.preview_images['metric_1'].config(text=str(len(findings)-likely))
        for ch in self.items.winfo_children():ch.destroy()
        self._finding_widgets={}
        if not findings:self.label(self.items,'No prominent changes found.',10,fg=GREEN).pack(anchor='w',padx=8,pady=9)
        for i,f in findings:
            box=tk.Frame(self.items,bg=PANEL2,cursor='hand2',highlightbackground=STROKE,highlightthickness=1)
            box.pack(fill='x',pady=4,padx=3)
            self._finding_widgets[i]=box
            line=tk.Frame(box,bg=PANEL2,cursor='hand2');line.pack(fill='x',padx=11,pady=(9,3))
            self.label(line,f'#{i}  {"Missing part" if f.review_status=="Likely missing" else "Changed area"}',10,bold=True,bg=PANEL2,wraplength=175,justify='left').pack(side='left')
            self.label(line,'MASKED' if f.occlusion_detected else ('MISSING' if f.review_status=='Likely missing' else 'CHECK'),8,
                       fg=BLUE if f.occlusion_detected else (RED if f.review_status=='Likely missing' else AMBER),bold=True,bg=PANEL2).pack(side='right')
            self.label(box,f'Position {f.x}, {f.y} · {f.w}×{f.h}px · Click to zoom',9,fg=MUTED,bg=PANEL2,wraplength=285,justify='left').pack(anchor='w',padx=11,pady=(0,10))
            def show(event, n=i, finding=f):self.show_finding(n,finding)
            for widget in [box]+list(box.winfo_children())+list(line.winfo_children()):widget.bind('<Button-1>',show)
        self.list_canvas.yview_moveto(0)
        self._update_scrollbar()
        warning=f'  ·  {len(result.quality_warnings or [])} image quality warning(s)' if result.quality_warnings else ''
        self.status.config(text=f'●  Complete · {likely} missing · {len(findings)-likely} changed (check){warning}')
        if result.quality_warnings:
            messagebox.showwarning('Image quality', '\n'.join(result.quality_warnings))

    def show_finding(self,number,finding):
        if self.result is None:return
        # The embedded viewer shows reference/aligned images in the same pixel coordinate system.
        # Never re-register or crop with a different transform at this stage.
        self.compare_viewer.open(self.result.reference,self.result.aligned,number,finding)
        self.result_preview.grid_remove()
        self.compare_viewer.grid()
        overlay=render_findings(self.result.aligned,self.result.findings,selected_id=number)
        self.result_preview.set(overlay)
        # Always reveal the selected location on BOTH source board previews.
        self.ref_preview.set(render_findings(self.result.reference,[finding],selected_id=1))
        self.test_preview.set(render_findings(self.result.aligned,[finding],selected_id=1))
        self.status.configure(text=f'●  Viewing finding #{number} · mouse wheel to zoom · drag to pan · Back to results to close.')

    def close_comparison(self):
        if not hasattr(self,'compare_viewer'):return
        self.compare_viewer.grid_remove()
        self.result_preview.grid()
        if getattr(self,'result',None) is not None:
            self.result_preview.set(self.result.overlay)
            self.ref_preview.set(self.result.reference)
            self.test_preview.set(self.result.aligned)
            self.status.configure(text='●  Inspection results · Select a finding to compare.')

    def _refresh_review_after_decision(self):
        if self.result is None:return
        self.result.review_findings=[f for f in self.result.findings
                                     if f.review_status!='Low-priority change']
        self.result.overlay=render_findings(self.result.aligned,self.result.findings)
        self._refresh_counts()
        self.close_comparison()
        for i,box in self._finding_widgets.items():
            box.pack_forget()
            if i<=len(self.result.findings) and self.result.findings[i-1].review_status!='Low-priority change':
                box.pack(fill='x',pady=4,padx=3)
        self._update_scrollbar()
        self.undo_btn.config(state='normal' if self._dismiss_history else 'disabled')

    def _refresh_counts(self):
        if self.result is None:return
        active=[f for f in self.result.findings if f.review_status!='Low-priority change']
        likely=sum(f.review_status=='Likely missing' for f in active)
        self.preview_images['metric_0'].config(text=str(likely))
        self.preview_images['metric_1'].config(text=str(len(active)-likely))
        self.status.configure(text=f'●  {likely} likely missing · {len(active)-likely} review · '
                                   'Not a defect removes a false alert; Undo restores it.')

    def dismiss_finding(self,number,finding):
        if self.result is None or finding.review_status=='Low-priority change':return
        old=(finding.review_status,finding.missing_likelihood,finding.user_decision)
        self._dismiss_history.append((finding,old))
        finding.review_status='Low-priority change'
        finding.missing_likelihood='User-marked not a defect'
        finding.user_decision='Not a defect'
        self._refresh_review_after_decision()

    def undo_dismissed(self):
        if not self._dismiss_history:return
        finding,old=self._dismiss_history.pop()
        finding.review_status,finding.missing_likelihood,finding.user_decision=old
        self._refresh_review_after_decision()

    def export(self):
        if self.result is None:return
        where=filedialog.askdirectory(title='Choose report destination')
        if not where:return
        try:
            destination=Path(where)/('PCB_Inspection_Report_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
            index=2
            while destination.exists():
                destination=Path(where)/('PCB_Inspection_Report_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{index}')
                index+=1
            path=save_report(self.result,destination)
            self.status.config(text='●  Report saved to '+str(path))
            messagebox.showinfo('Report saved',f'Images and JSON saved to:\n{path}')
        except Exception as exc:messagebox.showerror('Export failed',str(exc))

def main():
    App().mainloop()

if __name__=='__main__':main()
