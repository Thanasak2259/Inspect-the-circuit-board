"""High-contrast, minimum-size finding locators without altering true ROIs.

The physical ROI remains unchanged for analysis and reporting; marker halos
and badges are visual only. This works at small board export resolutions too.
"""
import cv2

COLORS={
    'Likely missing':(65,65,255),
    'Needs review':(40,196,255),
    'Low-priority change':(140,140,140),
}


def _mark(canvas,f,index,selected=False):
    H,W=canvas.shape[:2]
    vx,vy,vw,vh=getattr(f,'display_box',None) or (f.x,f.y,f.w,f.h)
    cx=int(vx+vw/2);cy=int(vy+vh/2)
    if f.review_status=='Low-priority change' or getattr(f,'user_decision','')=='Not a defect':
        return
    color=(210,105,245) if f.occlusion_detected else COLORS.get(f.review_status,(40,196,255))
    # Outer frame may grow, but actual measurement box remains true x,y,w,h.
    min_size=20 if selected else 14
    halfw=max(min_size//2,int(vw/2)+1)
    halfh=max(min_size//2,int(vh/2)+1)
    x1=max(0,cx-halfw);y1=max(0,cy-halfh)
    x2=min(W-1,cx+halfw);y2=min(H-1,cy+halfh)
    outer=4 if selected else 3
    cv2.rectangle(canvas,(x1,y1),(x2,y2),(15,9,28),outer+2,cv2.LINE_AA)
    cv2.rectangle(canvas,(x1,y1),(x2,y2),color,outer,cv2.LINE_AA)
    # Tight display geometry; the analysis/measurement ROI is not modified.
    cv2.rectangle(canvas,(vx,vy),(vx+vw,vy+vh),(255,255,255),1,cv2.LINE_AA)
    size=7 if not selected else 12
    cv2.line(canvas,(max(0,cx-size),cy),(min(W-1,cx+size),cy),(17,9,30),3,cv2.LINE_AA)
    cv2.line(canvas,(cx,max(0,cy-size)),(cx,min(H-1,cy+size)),(17,9,30),3,cv2.LINE_AA)
    cv2.drawMarker(canvas,(cx,cy),color,cv2.MARKER_CROSS,10 if not selected else 17,2,cv2.LINE_AA)
    # Pill badge within image boundaries; white index over dark fill.
    tag=f'#{index}'
    fontsize=.48 if min(H,W)>400 else .4
    (tw,th),base=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,fontsize,2)
    tw+=12;bh=th+11
    bx=min(max(0,x1),max(0,W-tw-1))
    by=(y1-bh-2 if y1>=bh+3 else min(H-bh-1,y2+3))
    by=max(0,by)
    cv2.rectangle(canvas,(bx,by),(bx+tw,by+bh),(22,12,35),-1)
    cv2.rectangle(canvas,(bx,by),(bx+tw,by+bh),color,2)
    cv2.putText(canvas,tag,(bx+6,by+th+5),cv2.FONT_HERSHEY_SIMPLEX,fontsize,(255,255,255),2,cv2.LINE_AA)


def render_findings(image,findings,selected_id=None):
    canvas=image.copy()
    for i,f in enumerate(findings,1):
        if i==selected_id:continue
        _mark(canvas,f,i)
    if selected_id is not None and 1<=selected_id<=len(findings):
        _mark(canvas,findings[selected_id-1],selected_id,selected=True)
    return canvas
