"""One reviewed geometry specification, rendered independently by TikZ and Matplotlib."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyArrowPatch,Arc,Rectangle
from shapely.geometry import Polygon as SPolygon

BASE=Path(__file__).resolve().parent
BLUE='#28649A';ORANGE='#C47820';INK='#26313A';GRAY='#78828B';PALE='#E8F1F8';GOLD='#F9E7CB'
plt.rcParams.update({'font.family':'Microsoft YaHei','mathtext.fontset':'stix','axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})

class Scene:
    def __init__(self,w,h):self.w=w;self.h=h;self.items=[]
    def line(self,pts,color=INK,width=.8,dash=False,arrow=None):self.items.append(('line',dict(pts=pts,color=color,width=width,dash=dash,arrow=arrow)))
    def polygon(self,pts,fill=PALE,color=BLUE,width=.8):self.items.append(('polygon',dict(pts=pts,fill=fill,color=color,width=width)))
    def text(self,x,y,label,size=9,color=INK,ha='center',va='center'):self.items.append(('text',dict(x=x,y=y,label=label,size=size,color=color,ha=ha,va=va)))
    def dot(self,x,y,color=INK,r=.038):self.items.append(('dot',dict(x=x,y=y,color=color,r=r)))
    def arc(self,x,y,r,a,b,color=INK,width=.7,arrow=None):
        angles=np.radians(np.linspace(a,b,101));self.line(np.c_[x+r*np.cos(angles),y+r*np.sin(angles)].tolist(),color,width,arrow=arrow)
    def export(self,name):
        self.geometry_only(name)
        folder=BASE/'python';folder.mkdir(exist_ok=True)
        fig=plt.figure(figsize=(self.w/2.54,self.h/2.54));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,self.w),ylim=(0,self.h),aspect='equal');ax.axis('off')
        tex=[r'\documentclass[tikz,border=0pt]{standalone}',r'\usepackage{amsmath,amssymb,fontspec,xeCJK}',r'\setmainfont{Times New Roman}',r'\setCJKmainfont{SimSun}',r'\usetikzlibrary{arrows.meta}',r'\begin{document}',r'\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]',f'\\path[use as bounding box] (0,0) rectangle ({self.w},{self.h});']
        def col(c):return '{rgb,255:red,'+str(int(c[1:3],16))+';green,'+str(int(c[3:5],16))+';blue,'+str(int(c[5:7],16))+'}'
        def coords(pts):return ' -- '.join(f'({x:.6f},{y:.6f})' for x,y in pts)
        for kind,d in self.items:
            if kind=='line':
                pts=np.asarray(d['pts']);arr=d['arrow']
                if arr and len(pts)==2:
                    ax.add_patch(FancyArrowPatch(pts[0],pts[1],arrowstyle=arr,mutation_scale=8,color=d['color'],lw=d['width'],linestyle='--' if d['dash'] else '-',shrinkA=0,shrinkB=0))
                else:
                    ax.plot(pts[:,0],pts[:,1],color=d['color'],lw=d['width'],ls='--' if d['dash'] else '-')
                    if arr:ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle=arr,mutation_scale=7,color=d['color'],lw=d['width'],shrinkA=0,shrinkB=0))
                opts=[f'draw={col(d["color"])}',f'line width={d["width"]}pt']
                if d['dash']:opts+=['dashed']
                if arr:opts+=[{'->':'-{Stealth[length=1.4mm]}','<->':'{Stealth[length=1.4mm]}-{Stealth[length=1.4mm]}'}[arr]]
                tex.append('\\draw['+','.join(opts)+'] '+coords(pts)+';')
            elif kind=='polygon':
                ax.add_patch(Polygon(d['pts'],closed=True,facecolor=d['fill'],edgecolor=d['color'],lw=d['width']))
                tex.append('\\path[fill='+col(d['fill'])+',draw='+col(d['color'])+f',line width={d["width"]}pt] '+coords(d['pts'])+' -- cycle;')
            elif kind=='dot':
                ax.add_patch(plt.Circle((d['x'],d['y']),d['r'],color=d['color']))
                tex.append(f'\\fill[fill={col(d["color"])}] ({d["x"]},{d["y"]}) circle ({d["r"]}cm);')
            else:
                ax.text(d['x'],d['y'],d['label'],fontsize=d['size'],color=d['color'],ha=d['ha'],va=d['va'])
                anchors={('left','center'):'west',('right','center'):'east',('center','center'):'center',('center','bottom'):'south',('center','top'):'north'}
                anchor=anchors.get((d['ha'],d['va']),'center')
                text=d['label'].replace('\n',r'\\')
                tex.append(f'\\node[anchor={anchor},align=center,text={col(d["color"])},font=\\fontsize{{{d["size"]}}}{{{d["size"]*1.25}}}\\selectfont,inner sep=0pt] at ({d["x"]},{d["y"]}) {{{text}}};')
        for ext in ['png','pdf','svg']:fig.savefig(folder/(name+'.'+ext),dpi=300,facecolor='white')
        plt.close(fig)
        tex.extend([r'\end{tikzpicture}',r'\end{document}'])
        (BASE/'rendered'/(name+'.tex')).write_text('\n'.join(tex),encoding='utf-8')

    def geometry_only(self,name):
        """Keep geometric annotations; explanations belong in the paper caption."""
        cleaned=[]
        for kind,original in self.items:
            d=original.copy()
            if kind=='text':
                label=d['label']
                if not (label.startswith('$') and label.endswith('$')):
                    continue
                if any(fragment in label for fragment in [r'\dfrac',r'D=\max',r'\qquad',r'\mathcal{F}_p(y)=']):
                    continue
                if label.startswith('$dA='):d['label']='$dA$'
                if label==r'$r=r_{\max}$':d['label']=r'$r_{\max}$'
            if kind in ['line','polygon']:
                pts=np.array(d['pts'],dtype=float)
                center=pts.mean(axis=0)
            else:center=np.array([d['x'],d['y']])
            if name=='fig_prior_sector':
                # Centre the enlarged differential element beside the sector.
                if kind in ['line','polygon']:
                    pts[:,1]-=.65
                    pts[pts[:,0]>=9.4,1]-=1.8
                else:
                    d['y']-=.65+(1.8 if d['x']>=9.4 else 0.)
                self.h=6.8
            else:
                # Global geometry left; two local examples stacked at right.
                shift=5.1 if center[0]<9.7 else (3.6 if center[1]>6 else 1.78)
                if kind in ['line','polygon']:pts[:,1]-=shift
                else:d['y']-=shift
                self.h=6.5
            if kind in ['line','polygon']:d['pts']=pts.tolist()
            cleaned.append((kind,d))
        self.items=cleaned

def polar(c,r,a):return np.asarray(c)+r*np.array([math.cos(math.radians(a)),math.sin(math.radians(a))])
def sector(c,r0,r1,a,b):
    return np.vstack([np.array([polar(c,r1,x) for x in np.linspace(a,b,161)]),np.array([polar(c,r0,x) for x in np.linspace(b,a,161)])]).tolist()

def prior():
    s=Scene(16,8.2);c=(1.05,3.85)
    s.text(4.2,7.65,'(a) 首次观测后的面积均匀先验',10.5)
    s.text(12.25,7.65,'(b) 面积微元的局部放大',10.5)
    s.polygon(sector(c,.8,6.,-24,24))
    s.line([[.45,c[1]],[7.85,c[1]]],GRAY,.65,arrow='->');s.text(7.98,3.84,'$x$',10)
    s.line([[c[0],.85],[c[0],6.9]],GRAY,.65,arrow='->');s.text(1.04,7.08,'$y$',10)
    s.dot(*c);s.text(.70,3.51,'$O_1$',10)
    for a in [-24,24]:s.line([c,polar(c,6.25,a)],BLUE,.85)
    s.text(6.83,6.58,r'$+\delta$',10,color=BLUE)
    s.text(6.83,1.12,r'$-\delta$',10,color=BLUE)
    s.text(4.78,2.62,r'$\Omega_1$',14,color=BLUE)
    s.text(4.85,3.49,r'$\widehat{\alpha}_1=0$',10,color=GRAY)
    s.text(2.45,4.95,r'$r_{\min}$',10);s.line([[2.12,4.88],polar(c,.8,19)],GRAY,.65)
    s.text(6.62,4.22,r'$r_{\max}$',10);s.line([[6.85,4.40],polar(c,6.,7)],GRAY,.65)
    s.arc(*c,1.75,0,11,GRAY,arrow='->');s.text(3.14,4.11,r'$\theta_1$',9)
    s.polygon(sector(c,3.9,4.5,11,18),GOLD,ORANGE,1.)
    for a in [11,18]:s.line([c,polar(c,4.65,a)],ORANGE,.6,True)
    s.text(4.28,4.24,r'$r_1$',10,color=ORANGE)
    s.line([polar(c,3.9,23),polar(c,4.5,23)],ORANGE,.7,arrow='<->');s.text(4.94,5.86,r'$dr_1$',9,color=ORANGE)
    s.line([polar(c,4.50,17),[9.45,6.15]],GRAY,.6,True,arrow='->')
    # Enlarged area element: local tangential and radial dimensions.
    s.polygon([[10,4.55],[14.35,4.55],[14.35,6.40],[10,6.40]],GOLD,ORANGE,1.)
    s.line([[10,4.15],[14.35,4.15]],ORANGE,.7,arrow='<->');s.text(12.18,3.83,r'$dr_1$',10,color=ORANGE)
    s.line([[14.75,4.55],[14.75,6.40]],ORANGE,.7,arrow='<->');s.text(14.80,5.47,r'$r_1\,d\theta_1$',9,color=ORANGE,ha='left')
    s.text(12.18,5.47,r'$dA=r_1\,dr_1\,d\theta_1$',12,color=ORANGE)
    s.text(12.16,3.05,r'$dP=\dfrac{r_1\,dr_1\,d\theta_1}{\delta(r_{\max}^2-r_{\min}^2)}$',11)
    s.text(12.1,1.93,r'$f_{r_1}(r_1)=\dfrac{2r_1}{r_{\max}^2-r_{\min}^2}$',10)
    s.text(12.1,1.07,r'$f_{\theta_1}(\theta_1)=\dfrac{1}{2\delta}$',10)
    s.text(8.,.42,'示意图：主图角度及内半径已放大；实际 '+r'$\delta=1^\circ$'+'，'+r'$r_{\min}=5\,\mathrm{m}$'+'，'+r'$r_{\max}=1500\,\mathrm{m}$',8,color=GRAY)
    s.export('fig_prior_sector')

def map_points(points,origin,scale):return (np.asarray(points)*scale+np.asarray(origin)).tolist()

def intersection():
    ex=json.loads((BASE/'geometry_examples.json').read_text());p=np.array(ex[0]['point']);target=np.array(ex[0]['target']);y=ex[0]['reading_rad'];alpha=math.pi/180
    s=Scene(17,11.3)
    s.text(4.8,10.75,'(a) 双站布设与真实测向几何',10.5)
    s.text(13.15,10.75,'(b) 未触及圆弧的四边形特例',10.5)
    # Global panel uses one scale for x and y.
    org=np.array([.8,6.9]);scale=.0054
    tr=lambda pts:map_points(pts,org,scale)
    s.polygon(tr(sector((0,0),5,1500,-1,1)),PALE,BLUE,.8)
    rayend=1200
    wpoly=[p,p+rayend*np.array([math.cos(y-alpha),math.sin(y-alpha)]),p+rayend*np.array([math.cos(y+alpha),math.sin(y+alpha)])]
    # The display-only wedge is cut at the bottom edge of the global panel.
    wedge=SPolygon(wpoly).intersection(SPolygon([[-100,-80],[1630,-80],[1630,1000],[-100,1000]]))
    s.polygon(tr(list(wedge.exterior.coords)),GOLD,ORANGE,.8)
    for sign in [-1,1]:s.line(tr([[0,0],[1520*math.cos(alpha),sign*1520*math.sin(alpha)]]),BLUE,.8)
    s.line(tr([[-60,0],[1610,0]]),GRAY,.6,arrow='->');s.text(9.65,6.88,'$x$',9)
    s.line(tr([[0,-100],[0,630]]),GRAY,.6,arrow='->');s.text(.78,10.49,'$y$',9)
    s.line(tr([[0,0],p]),INK,.9);s.text(2.43,8.50,'$L$',11)
    s.line(tr([[0,0],target]),GRAY,.65,True)
    s.line(tr([p,target]),INK,.8,True)
    q=tr([p])[0];s.line([q,[q[0]+1.5,q[1]]],GRAY,.55,True)
    s.arc(*q,.8,math.degrees(y),0,ORANGE,.7);s.text(q[0]+1.08,q[1]-.3,r'$\phi_2$',10,color=ORANGE)
    s.arc(*org,1.15,0,math.degrees(math.atan2(p[1],p[0])),INK,.7);s.text(2.18,7.49,r'$\gamma$',10)
    for point,label,offset in [([0,0],'$O_1$',[-.17,-.35]),(p,'$O_2$',[0,.32]),(target,'$T$',[0,.32])]:
        pt=np.array(tr([point])[0]);s.dot(*pt);s.text(*(pt+offset),label,10)
    s.text(4.84,5.96,r'$\widehat{\alpha}_1=0,\qquad \widehat{\alpha}_2=\phi_2+\varepsilon$',11)
    s.text(4.84,5.29,'蓝色：首次区域；橙色：第二次观测楔形',8.5)
    s.text(4.84,4.77,'全局主图按比例绘制；真实目标仅用于解释几何',8,color=GRAY)
    s.text(4.84,3.56,r'$\mathcal{F}_p(y)=\Omega_1\cap W(p,y,\delta)\setminus B(p,5)$',10.5)
    s.text(4.84,2.73,'保留距离约束，不预设交集恒为四边形',9)
    s.text(4.84,1.86,'首次读数已经固定；第二次读数尚未获得',8.5,color=GRAY)
    # Local quadrilateral example at r=1400 m.
    loc=np.array([11.55,8.40]);zoom=.025
    tl=lambda pts:map_points(np.asarray(pts)-[1400,0],loc,zoom)
    pts={z['label']:np.array([z['x'],z['y']]) for z in ex[0]['intersections']}
    order=['P11','P12','P22','P21']
    s.polygon(tl([pts[k] for k in order]),GOLD,ORANGE,1.)
    for a in [-1,1]:s.line(tl([[1330,a*1330*math.tan(alpha)],[1490,a*1490*math.tan(alpha)]]),BLUE,.75)
    for k1,k2 in [('P11','P21'),('P12','P22')]:
        v=pts[k2]-pts[k1];s.line(tl([pts[k1]-.18*v,pts[k2]+.18*v]),ORANGE,.75)
    s.line(tl([pts['P12'],pts['P21']]),INK,1.3,True)
    offsets={'P11':[-.10,-.30],'P12':[.15,-.30],'P21':[-.35,.30],'P22':[.20,.30]}
    for k,v in pts.items():
        pt=np.array(tl([v])[0]);s.dot(*pt,r=.029);s.text(*(pt+offsets[k]),'$'+k[0]+'_{'+k[1:]+'}$',9)
    s.text(14.50,8.42,r'$\mathcal{R}$',12,color=ORANGE)
    s.text(13.03,6.99,r'$D=\max_{i<j}\|V_i-V_j\|_2$',11)
    s.text(13.03,6.41,'图中四交点均满足射线及距离约束',8,color=GRAY)
    # Boundary case from the reviewed numerical counterexample.
    s.text(12.95,5.68,'(c) 1500 米边界参与裁剪',10.5)
    loc2=np.array([12.4,3.68]);z2=.025
    tc=lambda pts:map_points(np.asarray(pts)-[1500,0],loc2,z2)
    raw=[np.array([v['x'],v['y']]) for v in ex[1]['intersections']]
    poly=SPolygon([raw[i] for i in [0,1,3,2]])
    priorpoly=SPolygon(sector((0,0),5,1500,-1,1))
    clipped=poly.intersection(priorpoly)
    s.polygon(tc(list(clipped.exterior.coords)),GOLD,ORANGE,1.)
    s.line(tc([raw[i] for i in [0,1,3,2,0]]),GRAY,.75,True)
    arc=np.c_[1500*np.cos(np.linspace(-.023,.024,241)),1500*np.sin(np.linspace(-.023,.024,241))]
    s.line(tc(arc),BLUE,1.2)
    s.text(11.40,3.24,r'$\mathcal{F}_p(y)$',10,color=ORANGE)
    s.line([[11.52,3.50],tc([[1492.,23.]])[0]],ORANGE,.65)
    s.text(14.44,2.68,r'$r=r_{\max}$',9,color=BLUE);s.line([[13.99,2.88],tc([[1499.8,-18]])[0]],BLUE,.6)
    s.text(13.02,2.03,'虚线四边形包含不满足距离约束的点',8,color=GRAY)
    s.text(13.03,1.43,'未裁剪：130.99 m；裁剪后：32.88 m',9)
    s.text(8.5,.42,'两幅局部图均等比例放大；'+r'$\delta=1^\circ$'+'。一般区域应取扇形、楔形与距离约束的完整交集。',8,color=GRAY)
    s.export('fig_geometry_intersection')

if __name__=='__main__':prior();intersection();print('Two figures exported through Matplotlib; TikZ sources prepared.')
