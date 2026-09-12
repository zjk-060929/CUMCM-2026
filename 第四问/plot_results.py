"""Generate publication-exportable figures from saved experiment data."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle,Wedge
from geometry import DEFAULT_SIDE,triangular_mesh,open_route

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
available={f.name for f in font_manager.fontManager.ttflist}
font=next((f for f in ['Arial Unicode MS','Heiti SC','Songti SC','SimHei','Noto Sans CJK SC'] if f in available),'DejaVu Sans')
plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                     'savefig.facecolor':'white','svg.fonttype':'none'})
BLUE='#2463a6'; ORANGE='#d78024'; GREEN='#28876b'; GREY='#b9c5ce'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def coverage():
    points,triangles=triangular_mesh(); points=np.array(points)
    fig,ax=plt.subplots(figsize=(9,8))
    for tri in triangles:
        p=np.array((*tri,tri[0])); ax.plot(p[:,0],p[:,1],color=GREY,lw=.75,zorder=0)
    ax.add_patch(Circle((0,0),1800,fill=False,lw=2,color=BLUE,label='目标圆域（半径 1800 m）'))
    outer=np.linalg.norm(points,axis=1)>1800
    ax.scatter(*points[~outer].T,s=38,color=BLUE,label='圆域内检测站')
    ax.scatter(*points[outer].T,s=45,color=ORANGE,label='圆域外检测站（不可随意删除）')
    ax.add_patch(Wedge((1800,0),1000,-90,90,facecolor=GREEN,alpha=.12))
    ax.scatter([1800],[0],marker='*',s=150,color=GREEN,zorder=5)
    ax.annotate('边界处向外辐射的示意源',xy=(1800,0),xytext=(800,-1250),
                arrowprops={'arrowstyle':'->','color':GREEN},color=GREEN)
    ax.set_title(f'31 个站点 · 42 个三角形 · 边长 {DEFAULT_SIDE:g} m',fontsize=14,pad=15)
    ax.set(xlim=(-2800,2900),ylim=(-2800,2800),xlabel='东向坐标 x / m',ylabel='北向坐标 y / m',aspect='equal')
    ax.legend(loc='upper left',bbox_to_anchor=(0,.93),frameon=True,fontsize=10)
    ax.grid(alpha=.15); fig.tight_layout(); save(fig,'01_方向覆盖')

def comparisons():
    with (ROOT/'results/paired_cases.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    keys=['square_optical','square_active','triangle_optical','triangle_active']
    labels=['方格＋光学','方格＋主动','三角＋光学','三角＋主动']
    groups=[[r for r in rows if r['variant']==k] for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    data=[[float(r['virtual_time_s'])/60 for r in g] for g in groups]
    bp=axes[0].boxplot(data,tick_labels=labels,patch_artist=True,showfliers=False,medianprops={'color':'white','lw':2})
    for patch,color in zip(bp['boxes'],[GREY,BLUE,ORANGE,GREEN]): patch.set_facecolor(color)
    axes[0].set_ylabel('总虚拟时间 / min'); axes[0].set_title('500 组配对场景的耗时分布')
    axes[0].grid(axis='y',alpha=.2)
    movement=[np.mean([float(r['distance_m'])/5/60 for r in g]) for g in groups]
    detection=[np.mean([(float(r['measures'])*5+float(r['switches']))/60 for r in g]) for g in groups]
    clearing=[np.mean([(float(r['cleared'])*5+float(r['clear_failures'])*3)/60 for r in g]) for g in groups]
    x=np.arange(4)
    axes[1].bar(x,movement,color=BLUE,label='移动')
    axes[1].bar(x,detection,bottom=movement,color=ORANGE,label='测量＋切频')
    axes[1].bar(x,clearing,bottom=np.array(movement)+detection,color=GREEN,label='光学定位＋清除')
    for i in x: axes[1].text(i,movement[i]+detection[i]+clearing[i]+4,f'{np.mean(data[i]):.1f}',ha='center',fontsize=10)
    axes[1].set_xticks(x,labels); axes[1].set_ylabel('平均虚拟时间 / min'); axes[1].set_title('耗时分解（包含收尾搜索）')
    axes[1].legend(loc='upper right'); axes[1].set_ylim(0,380)
    fig.suptitle('自建模拟实验：四种策略均为 500 / 500 组全部清除',fontsize=14,y=1.02)
    fig.tight_layout(); save(fig,'02_策略对比')

def trajectory():
    folder=ROOT/'results/sample_seed_42'
    truth=json.loads((folder/'truth_after_exit.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    actions=[json.loads(x) for x in (folder/'actions.jsonl').read_text().splitlines()]
    path=[(0.,0.)]; clears=[]
    for record in actions:
        if record['event']=='request' and 'position' in record['body']:
            p=record['body']['position']; point=(p['x'],p['y'])
            if point!=path[-1]: path.append(point)
        if record['event']=='response' and record['path']=='/clear' and record['body']['clear_result']=='success': clears.append(path[-1])
    path=np.array(path); clears=np.array(clears)
    fig,ax=plt.subplots(figsize=(9,8))
    ax.add_patch(Circle((0,0),1800,fill=False,color=BLUE,lw=1.5))
    ax.plot(path[:,0],path[:,1],color=GREY,lw=1.2,label='实际行走轨迹',zorder=1)
    p=np.array(triangular_mesh()[0]); ax.scatter(*p.T,color=BLUE,s=18,label='覆盖站点')
    for s in truth:
        if s['orientation'] is not None:
            angle=math.radians(s['orientation'])
            ax.arrow(s['x'],s['y'],180*math.cos(angle),180*math.sin(angle),width=6,head_width=50,color=ORANGE,zorder=4)
        ax.scatter([s['x']],[s['y']],marker='*',s=100,color=ORANGE,zorder=4)
        ax.annotate(f"C{s['channel']}",(s['x'],s['y']),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.scatter(*clears.T,s=75,facecolors='none',edgecolors=GREEN,lw=1.5,label='成功清除点',zorder=5)
    ax.scatter([0],[0],marker='s',s=55,color='black',label='起点')
    ax.set(xlabel='x / m',ylabel='y / m',aspect='equal',xlim=(-2800,2800),ylim=(-2800,2800))
    ax.set_title(f"自建案例 seed = 42：10 / 10 清除，总虚拟时间 {summary['virtual_time_s']:.3f} s",pad=15)
    ax.legend(loc='upper left',fontsize=10); ax.grid(alpha=.12)
    fig.text(.5,.01,'星形和朝向箭头为退出后揭示的真值；算法运行时无法读取。',ha='center',fontsize=10,color='#586773')
    fig.tight_layout(rect=(0,.025,1,1)); save(fig,'03_样例轨迹')

def stress():
    summary=json.loads((ROOT/'results/benchmark_summary.json').read_text())['stress']
    labels=['边界向外／+1°','边界向外／−1°','最小半径／平滑误差','最小半径／跳变误差',
            '局部密集／+1°','近原点／固定位置误差','全定向／跳变（扩展）','全向／平滑（扩展）']
    means=[s['virtual_time_s']['mean']/60 for s in summary.values()]
    maxima=[s['virtual_time_s']['max']/60 for s in summary.values()]
    fig,ax=plt.subplots(figsize=(10,6))
    y=np.arange(8); ax.barh(y,means,color=[GREEN]*6+[GREY]*2,label='平均耗时')
    ax.scatter(maxima,y,marker='D',color=ORANGE,s=40,label='最大耗时',zorder=4)
    for i,maximum in enumerate(maxima): ax.text(maximum+4,i,'50 / 50',va='center',fontsize=10)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,300)
    ax.set_xlabel('总虚拟时间 / min'); ax.set_title('400 组压力测试：每组均全部清除（自建模拟）')
    ax.legend(loc='lower right'); ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); save(fig,'04_压力测试')

if __name__=='__main__':
    coverage(); comparisons(); trajectory(); stress()
    print(f'Figures saved to {OUT}')
