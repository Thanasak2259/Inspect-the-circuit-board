"""Embedded, synchronized PCB reference vs inspection comparison viewer.

All coordinates operate in the reference-board pixel space. The displayed
pixels are cropped directly from original images, not enlarged 40px thumbnails.
"""
import tkinter as tk
from PIL import Image, ImageTk
import cv2

BG='#130724'; PANEL='#201238'; PANEL2='#2d1b49'; STROKE='#574176'
MUTED='#b7a5d1'; TEXT='#f9f2ff'; BLUE='#a878ff'; RED='#ff7fbc'; AMBER='#ffce8b'
FONT='Segoe UI'


def centered_crop_bounds(width,height,cx,cy,viewport_w,viewport_h,zoom):
    """Visible image rectangle at a zoom value in image-pixels to screen-pixels."""
    if width<=0 or height<=0 or viewport_w<=0 or viewport_h<=0 or zoom<=0:
        raise ValueError('Invalid image dimensions or zoom')
    view_w=min(width,viewport_w/zoom)
    view_h=min(height,viewport_h/zoom)
    left=max(0.0,min(width-view_w,cx-view_w/2))
    top=max(0.0,min(height-view_h,cy-view_h/2))
    return left,top,left+view_w,top+view_h


class ComparisonViewer(tk.Frame):
    """Inline pair of zoomable canvases with a shared viewport and drag pan."""
    def __init__(self,parent,on_back,on_dismiss=None):
        super().__init__(parent,bg=PANEL,highlightbackground=STROKE,highlightthickness=1)
        self.on_back=on_back
        self.on_dismiss=on_dismiss
        self.finding_number=None
        self.reference=None;self.aligned=None;self.finding=None
        self.center=(0.0,0.0);self.zoom=4.0;self._photos=[None,None]
        self._pan_anchor=None
        top=tk.Frame(self,bg=PANEL);top.pack(fill='x',padx=12,pady=(9,5))
        self.head=tk.Label(top,text='Finding comparison',fg=TEXT,bg=PANEL,font=(FONT,11,'bold'))
        self.head.pack(side='left')
        self._button(top,'Back to results',self.on_back).pack(side='right',padx=(7,0))
        if on_dismiss is not None:
            self._button(top,'Not a defect',self._dismiss).pack(side='right',padx=(7,0))
        self._button(top,'Center',self.center_finding).pack(side='right',padx=(4,0))
        self._button(top,'Fit',self.fit).pack(side='right',padx=(4,0))
        self._button(top,'+',lambda:self.change_zoom(1.5)).pack(side='right',padx=(4,0))
        self._button(top,'−',lambda:self.change_zoom(1/1.5)).pack(side='right',padx=(4,0))
        self.zoom_text=tk.Label(top,text='4×',fg=MUTED,bg=PANEL,font=(FONT,10))
        self.zoom_text.pack(side='right',padx=(4,8))
        pair=tk.Frame(self,bg=PANEL);pair.pack(fill='both',expand=True,padx=8,pady=(0,4))
        pair.grid_columnconfigure((0,1),weight=1,uniform='pair');pair.grid_rowconfigure(0,weight=1)
        self.canvases=[]
        for i,name in enumerate(('REFERENCE','INSPECTION · ALIGNED')):
            frame=tk.Frame(pair,bg=PANEL2,highlightbackground=STROKE,highlightthickness=1)
            frame.grid(row=0,column=i,sticky='nsew',padx=4)
            tk.Label(frame,text=name,fg=TEXT,bg=PANEL2,font=(FONT,9,'bold')).pack(anchor='w',padx=9,pady=(7,3))
            canvas=tk.Canvas(frame,bg='#10091f',highlightthickness=0,bd=0,cursor='fleur')
            canvas.pack(fill='both',expand=True,padx=7,pady=(0,7))
            canvas.bind('<Configure>',lambda e:self.render())
            canvas.bind('<MouseWheel>',self._wheel)
            canvas.bind('<Button-4>',lambda e:self.change_zoom(1.25))
            canvas.bind('<Button-5>',lambda e:self.change_zoom(.8))
            canvas.bind('<ButtonPress-1>',self._start_pan)
            canvas.bind('<B1-Motion>',self._do_pan)
            canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'_pan_anchor',None))
            self.canvases.append(canvas)
        self.footer=tk.Label(self,text='Select a finding to compare.',fg=MUTED,bg=PANEL,font=(FONT,9),anchor='w')
        self.footer.pack(fill='x',padx=12,pady=(0,9))
        self.bind('<Key-plus>',lambda e:self.change_zoom(1.5))
        self.bind('<Key-minus>',lambda e:self.change_zoom(1/1.5))

    def _button(self,parent,label,cmd):
        return tk.Button(parent,text=label,command=cmd,font=(FONT,9,'bold'),fg=TEXT,bg=PANEL2,
                         activeforeground=TEXT,activebackground='#493265',bd=0,relief='flat',
                         padx=10,pady=5,cursor='hand2')

    def open(self,reference,aligned,number,finding):
        self.finding_number=number
        self.reference=reference;self.aligned=aligned;self.finding=finding
        self.head.configure(text=f'#{number}  ·  {finding.component_type}  ·  '+
            ('MASKED / UNVERIFIED' if finding.occlusion_detected else finding.review_status))
        self.center_finding()
        self.footer.configure(text=f'Position: {finding.x}, {finding.y} px  ·  Mouse wheel: zoom  ·  Drag: pan  ·  Both views synchronized')
        self.after_idle(self.center_finding)  # calculate after grid becomes visible

    def _dismiss(self):
        if self.on_dismiss is not None and self.finding is not None:
            self.on_dismiss(self.finding_number,self.finding)

    def _shape(self):
        h,w=self.reference.shape[:2]
        return w,h

    def _fit_zoom(self):
        w,h=self._shape()
        cw=max(1,min(c.winfo_width() for c in self.canvases)-12)
        ch=max(1,min(c.winfo_height() for c in self.canvases)-12)
        return max(.03,min(cw/w,ch/h))

    def center_finding(self):
        if self.finding is None:return
        f=self.finding
        vx,vy,vw,vh=getattr(f,'display_box',None) or (f.x,f.y,f.w,f.h)
        self.center=(vx+vw/2,vy+vh/2)
        # Show a meaningful patch of board around the part, NOT a tiny center thumbnail.
        # The context spans roughly 3x the component bounding box while keeping landmarks visible.
        span=max(28,float(max(vw,vh))*3.0)
        cw=max(1,min(c.winfo_width() for c in self.canvases)-12)
        ch=max(1,min(c.winfo_height() for c in self.canvases)-12)
        self.zoom=max(self._fit_zoom(),min(18.,cw/span,ch/span))
        self.render()

    def fit(self):
        if self.reference is None:return
        w,h=self._shape();self.center=(w/2,h/2)
        self.zoom=self._fit_zoom()
        self.render()

    def change_zoom(self,factor):
        if self.reference is None:return
        self.zoom=max(self._fit_zoom(),min(24.,self.zoom*factor))
        self.render()

    def _wheel(self,event):
        self.change_zoom(1.25 if event.delta>0 else .8)
        return 'break'

    def _start_pan(self,event):
        self._pan_anchor=(event.x_root,event.y_root,self.center)

    def _do_pan(self,event):
        if self._pan_anchor is None or self.reference is None:return
        sx,sy,(cx,cy)=self._pan_anchor
        w,h=self._shape()
        self.center=(max(0,min(w,cx-(event.x_root-sx)/self.zoom)),
                     max(0,min(h,cy-(event.y_root-sy)/self.zoom)))
        self.render()

    def render(self):
        if self.reference is None:return
        for i,(canvas,arr) in enumerate(zip(self.canvases,(self.reference,self.aligned))):
            width,height=canvas.winfo_width(),canvas.winfo_height()
            if width<20 or height<20:continue
            cw,ch=width-12,height-12
            image_h,image_w=arr.shape[:2]
            # Even at fractional zoom, crop with a stable mapping between the two views.
            x1,y1,x2,y2=centered_crop_bounds(image_w,image_h,*self.center,cw,ch,self.zoom)
            left,top=int(x1),int(y1)
            right,bottom=min(image_w,max(left+1,int(x2+0.999))),min(image_h,max(top+1,int(y2+0.999)))
            crop=arr[top:bottom,left:right]
            rw=max(1,min(cw,int(round((right-left)*self.zoom))))
            rh=max(1,min(ch,int(round((bottom-top)*self.zoom))))
            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
            resample=Image.Resampling.NEAREST if self.zoom>=5 else Image.Resampling.BILINEAR
            photo=ImageTk.PhotoImage(Image.fromarray(rgb).resize((rw,rh),resample))
            self._photos[i]=photo
            canvas.delete('all')
            ox=(width-rw)/2;oy=(height-rh)/2
            canvas.create_image(ox,oy,image=photo,anchor='nw')
            f=self.finding
            if f is not None:
                vx,vy,vw,vh=getattr(f,'display_box',None) or (f.x,f.y,f.w,f.h)
                fx=ox+(vx-left)*rw/(right-left);fy=oy+(vy-top)*rh/(bottom-top)
                fw=vw*rw/(right-left);fh=vh*rh/(bottom-top)
                midx=fx+fw/2;midy=fy+fh/2
                # A larger visual locator helps with 0402/0603 parts even when
                # their true pixel footprint is only a few pixels wide.
                mark=BLUE if f.occlusion_detected else (RED if f.review_status=='Likely missing' else AMBER)
                halfw=max(13,fw/2+3);halfh=max(13,fh/2+3)
                xa,xb=midx-halfw,midx+halfw
                ya,yb=midy-halfh,midy+halfh
                canvas.create_rectangle(xa-2,ya-2,xb+2,yb+2,outline='#10091f',width=6)
                canvas.create_rectangle(xa,ya,xb,yb,outline=mark,width=3)
                # True component crop: thin solid line, NOT the enlarged ring.
                canvas.create_rectangle(fx,fy,fx+fw,fy+fh,outline='#ffffff',width=1)
                canvas.create_line(midx-15,midy,midx+15,midy,fill='#160a27',width=5)
                canvas.create_line(midx,midy-15,midx,midy+15,fill='#160a27',width=5)
                canvas.create_line(midx-14,midy,midx+14,midy,fill=mark,width=2)
                canvas.create_line(midx,midy-14,midx,midy+14,fill=mark,width=2)
                canvas.create_oval(midx-5,midy-5,midx+5,midy+5,outline='white',width=2)
                tx=max(8,min(width-120,xa));ty=max(4,ya-32)
                canvas.create_rectangle(tx,ty,tx+112,ty+23,fill='#10091f',outline=mark,width=2)
                canvas.create_text(tx+56,ty+12,text=f'TARGET #{self.finding_number}',
                                   fill='white',font=(FONT,10,'bold'))
        self.zoom_text.configure(text=f'{self.zoom:.1f}×')
