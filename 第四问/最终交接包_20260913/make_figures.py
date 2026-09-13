"""Publication PNG/SVG figures from archived, audited Q4 tables.

Chart contract: see 05_图表与数据字典.md. No simulations or new estimates.
Absolute means: zero-based horizontal bars, focal blue and neutral references.
Uncertainty: paired-saving point intervals, zero reference, 14 layout clusters.
Composition: stacked paid costs, per-case source-normalized before averaging.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
BLUE='#4C78A8'; INK='#30343B'; GREY='#B9BEC5'; GRID='#E4E6E9'
LABELS={'fifth':'历史第五轮','eighth':'原第八轮','boundary':'主线迭代 (2)',
        'reliable':'自主启发式可靠版','integrated':'本次整合版'}

def table(name):
    with (ROOT/'tables'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def setup():
    for name in ['Microsoft YaHei','Noto Sans CJK SC','SimHei','PingFang SC']:
        try:font_manager.findfont(name,fallback_to_default=False);break
        except ValueError:continue
    else:raise RuntimeError('A CJK font is required to regenerate Chinese figures; supplied PNG/SVG can be used directly.')
    plt.rcParams.update({'font.family':name,'font.size':11,'axes.unicode_minus':False,
                         'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
                         'axes.edgecolor':'#7F858D','axes.linewidth':.8,'svg.fonttype':'path',
                         'savefig.facecolor':'white','figure.facecolor':'white','hatch.linewidth':.6})

def scaffold(ax,axis='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis=axis,color=GRID,linewidth=.7)

def header(fig,title,subtitle):
    fig.text(.04,.955,title,fontsize=16,weight='bold',va='top')
    fig.text(.04,.89,subtitle,fontsize=10,va='top')

def save(fig,name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=250)
    fig.savefig(folder/(name+'.svg'),metadata={'Date':None})
    plt.close(fig)

def main():
    setup()
    data=table('ordinary_distribution.csv')
    fig,ax=plt.subplots(figsize=(9.2,5.3));fig.subplots_adjust(left=.24,right=.94,bottom=.16,top=.79)
    values=[float(r['mean_seconds_per_source']) for r in data]
    bars=ax.barh(range(5),values,height=.56,color=[BLUE if r['variant']=='integrated' else GREY for r in data],
                 edgecolor=INK,linewidth=.7)
    bars[-1].set_hatch('///')
    ax.set_yticks(range(5),[LABELS[r['variant']] for r in data]);ax.invert_yaxis()
    ax.set_xlim(0,520);ax.set_xlabel('平均定位清除时间（秒／源）')
    for i,v in enumerate(values):ax.text(v+6,i,f'{v:.2f}',va='center',fontsize=11)
    ax.axvline(400,color=INK,linestyle='--',linewidth=1.)
    ax.text(397,-.5,'400秒目标',ha='right',va='bottom',fontsize=9)
    scaffold(ax)
    header(fig,'普通场景平均定位清除时间','14个新布局 × 3种空间误差尺度 = 42场；每场先按源数归一，再平均')
    fig.text(.04,.035,'来源：冻结后的统一自建仿真；五方案均完成清除。程序计算时间另计。',fontsize=9)
    save(fig,'fig1_ordinary_mean')

    data=table('paired_improvement.csv')
    fig,ax=plt.subplots(figsize=(9.2,5.3));fig.subplots_adjust(left=.25,right=.80,bottom=.17,top=.79)
    for i,r in enumerate(data):
        m,lo,hi=[float(r[k]) for k in ['saved_seconds_per_source','ci95_low','ci95_high']]
        ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=BLUE,markeredgecolor=INK,
                    markerfacecolor=BLUE if lo>0 else 'white',markersize=7,capsize=5,lw=1.5)
        ax.text(1.03,i,f'{m:.2f}\n[{lo:.2f}, {hi:.2f}]',transform=ax.get_yaxis_transform(),
                va='center',ha='left',fontsize=10)
    ax.set_yticks(range(4),[LABELS[r['comparator']] for r in data]);ax.invert_yaxis()
    ax.set_xlim(-8,50);ax.set_xlabel('整合版相对对照的平均节省（秒／源；正值为更快）')
    ax.axvline(0,color=INK,linestyle='--',linewidth=1);scaffold(ax)
    header(fig,'普通场景配对时间改善及95%区间','14个独立布局聚合3种尺度；5000次布局配对自助重采样')
    fig.text(.04,.035,'来源：同场景原始记录。空心点所对应的区间包含0，尚不能确认稳定优势。',fontsize=9)
    save(fig,'fig2_paired_intervals')

    rows={r['variant']:r for r in table('paid_time_components.csv')}
    variants=['eighth','integrated'];bottom=[0.,0.]
    keys=['movement','measurement','switching','successful_clear','failed_clear']
    names=['移动','检测','换频','成功清除','失败清除']
    colors=[BLUE,'#E7BA52','#B8BABC','#B6C78C','#D8A2B3']
    hatches=['','///','xx','..','\\\\']
    fig,ax=plt.subplots(figsize=(9.2,5.6));fig.subplots_adjust(left=.14,right=.68,bottom=.15,top=.79)
    for key,name,color,hatch in zip(keys,names,colors,hatches):
        values=[float(rows[v][key+'_seconds_per_source']) for v in variants]
        ax.bar(range(2),values,bottom=bottom,width=.5,label=name,color=color,hatch=hatch,edgecolor=INK,lw=.6)
        for i,val in enumerate(values):
            if val>=20:ax.text(i,bottom[i]+val/2,f'{val:.1f}',ha='center',va='center',fontsize=11,
                              color='white' if key=='movement' else INK)
            bottom[i]+=val
    for i,total in enumerate(bottom):ax.text(i,total+8,f'{total:.2f}',ha='center',fontsize=11,weight='bold')
    ax.set_xticks(range(2),[LABELS[v] for v in variants]);ax.set_ylabel('平均执行时间（秒／源）')
    ax.set_ylim(0,530);scaffold(ax,'y');ax.legend(loc='center left',bbox_to_anchor=(1.04,.65),frameon=False)
    header(fig,'普通场景模拟执行时间组成','普通42场；各项费用在每场按源数归一后再平均，组成之和等于总均值')
    fig.text(.04,.035,'来源：逐场路程、检测、换频和清除账本；费用见正文式(1)。程序计算时间另计。',fontsize=9)
    save(fig,'fig3_paid_time')
    print('Wrote three publication figures in PNG and SVG.')

if __name__=='__main__':main()
