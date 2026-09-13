"""Audit and report completed common-simulator integration experiments."""
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics as st

Q4=Path(__file__).resolve().parent
ROOT=Q4/'integrated_q4_20260913'
LABELS={'fifth':'历史第五轮','eighth':'原第八轮','boundary':'主线迭代 (2)：边界补救',
        'reliable':'自主启发式：逐动作可靠版','integrated':'本次整合版','expanded':'整合＋扩大搜索',
        'free':'整合但取消巡检保护','no_sensing':'整合但关闭额外补测'}
MODES={'mixed':'普通','edge_outward':'边界朝外','rotated_cluster':'旋转聚集',
       'minimum_range':'最小接收半径','edge_inset':'边界内侧20～100米朝外'}


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(data,before,after):
    a={(r['seed'],r['length_scale_m']):r for r in data if r['variant']==before}
    b={(r['seed'],r['length_scale_m']):r for r in data if r['variant']==after}
    assert a.keys()==b.keys()
    diffs={k:a[k]['time_per_source_s']-b[k]['time_per_source_s'] for k in a}
    clusters=[st.mean(d for (s,_),d in diffs.items() if s==seed) for seed in sorted({s for s,_ in a})]
    rng=random.Random(612031)
    boot=sorted(st.mean(rng.choices(clusters,k=len(clusters))) for _ in range(5000))
    return dict(mean_saved_s=st.mean(diffs.values()),reduction_pct=100*st.mean(diffs.values())/st.mean(r['time_per_source_s'] for r in a.values()),
                layout_bootstrap_95ci=[boot[125],boot[4874]],layouts=len(clusters),cases=len(a),
                faster=sum(v>1e-6 for v in diffs.values()),slower=sum(v< -1e-6 for v in diffs.values()),ties=sum(abs(v)<=1e-6 for v in diffs.values()),
                worst_regressions=sorted([dict(seed=s,scale=l,extra_total_s=b[s,l]['virtual_time_s']-a[s,l]['virtual_time_s'],
                    before_s=a[s,l]['time_per_source_s'],after_s=b[s,l]['time_per_source_s'],before_stations=a[s,l]['stations_visited'],
                    after_stations=b[s,l]['stations_visited']) for s,l in a],key=lambda r:-r['extra_total_s'])[:3])


def main():
    config=load(ROOT/'results/frozen_config.json');selected=config['selected_variant']
    assert all(digest(ROOT/n)==h and digest(ROOT/'results/frozen_source'/n)==h for n,h in config['source_sha256'].items())
    stages={};allrows=[]
    for name in ['smoke','development','heldout']:
        stage=ROOT/'results'/name;manifest=load(stage/'manifest.json')
        data=[r for v in manifest['variants'] for r in rows(stage/f'{v}.jsonl')]
        assert len(data)==len(manifest['cases'])*len(manifest['variants'])
        assert len({(r['variant'],r['seed'],r['mode'],r['length_scale_m']) for r in data})==len(data)
        assert all(r['completed'] and r['clear_ratio']==1 and r['containment_violations']==r['nearby_effective_bearings']==0 for r in data)
        stages[name]=dict(runs=len(data),complete=len(data),source_sha256=manifest['source_sha256'])
        allrows+=data
        if name=='heldout':held=data
    entry=ROOT/'results/entry_verification/summary.json'
    extra=1 if entry.exists() else 0
    if extra:assert load(entry)['completed'] and load(entry)['clear_ratio']==1
    assert len(allrows)+extra<=550
    stats=load(ROOT/'results/heldout/summary.json')['groups']
    dev=load(ROOT/'results/development/summary.json')['groups']
    comparisons={m:{v:paired([r for r in held if r['mode']==m],v,selected)
                       for v in ['fifth','eighth','boundary','reliable']} for m in stats}
    audit=dict(stages=stages,complete_simulations=len(allrows)+extra,entry_runs=extra,
               focused_tests_passed=14,new_validation_layouts=len({(r['seed'],r['mode']) for r in held}),
               validation_cases_per_policy=len(held)//5,validation_runs=len(held),
               containment_violations=0,nearby_effective_bearings=0,selected=selected,
               selected_single_action_decisions=sum(r['decisions'] for r in held if r['variant']==selected),
               selected_single_action_replans=sum(r['schedule_replans'] for r in held if r['variant']==selected),
               comparisons=comparisons)
    assert audit['selected_single_action_decisions']==audit['selected_single_action_replans']
    for row in allrows:
        if row['variant'] not in ['integrated','expanded','free','no_sensing']:continue
        assert row['optional_count']<=40
        for check in row['guard_records']:
            assert check['candidate_station_m']>check['old_cap_m']+1e-5
            assert check['known_service_gain_s']>0
            if check['accepted']:assert check['paid_gain_s']-check['discovery_risk_s']>=20.
    (ROOT/'results/final_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    fields=['variant','seed','mode','length_scale_m','source_count','completed','time_per_source_s','virtual_time_s',
            'distance_m','measures','switches','clear_failures','fallback_attempts','stations_visited','runtime_s',
            'mean_first_discovery_s','last_first_discovery_s','containment_violations','nearby_effective_bearings']
    with (ROOT/'results/heldout/comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in held)
    p=comparisons['mixed']['eighth'];a=stats['mixed']['eighth'];b=stats['mixed'][selected]
    lines=['# 三个对话方法整合：统一误差场下的有限仿真','',
      f"本次冻结整合版的普通场景平均为 **{b['time_per_source_s']:.2f}秒/源**；同场原第八轮为{a['time_per_source_s']:.2f}秒/源，平均节省{p['mean_saved_s']:.2f}秒/源（{p['reduction_pct']:.2f}%）。该平均节省的按布局配对自助法95%区间为[{p['layout_bootstrap_95ci'][0]:.2f}, {p['layout_bootstrap_95ci'][1]:.2f}]。",'',
      '## 学习来源和实际改动','',
      '- 主线迭代 (2)：保留1024边保守定位，吸收可行圆周弧试清除和外侧60米、横向±90米交会；将原来串行的一整个补救过程拆成逐动作状态，允许每一步后重新规划。',
      '- 自主启发式算法：采用逐请求全局规划、未完成站点频道账本、当前位置与途中机会测量、连续区域光学风险和低收益补测暂停；修复仅在规划中评估兜底就消耗覆盖点或提前启用兜底状态的问题。',
      '- 主线迭代 (3)：参考巡检超限需要正服务收益、完整预计付费时间节省和最后发现风险保护的三项条件；保留真正需要超限才进入例外的检查。',
      '- 新整合：用拟执行的测量或清除位置作为路线入口，用预计服务出口接后续路线；实际执行规划选定动作。测量信息价值与下一终点观测比较，共享固定发射半径和朝向假设，不把收不到也算成缩区。',
      '- 已明确测试更大搜索：从3层/宽4增加到5层/宽12，并增加局部改进次数。开发普通均值略差，默认保留标准搜索，但逐动作规划本身已增加计算量；扩大搜索通过入口的--variant expanded保留。','',
      '更细圆近似、更大搜索或更多测量，都不是必然提速。完整误差界、未知频道覆盖与有限光学清除仍作为保证层。技术说明见本目录README.md。','',
      '## 比较方法','',
      f"本轮共{audit['complete_simulations']}次完整任务仿真：已知案例检查6次，开发154次，冻结后新布局检验345次，入口验证{extra}次；低于预设550次上限。另有14项针对性测试，不把局部协议会话计为完整任务。",'',
      '所有版本使用同一目标布局、同一四波平滑空间误差场及100、300、900米合成尺度；开发仅取100、900米。自主启发式原成绩使用过另一种平滑场，不能直接将其448.10与其他分支的453秒排名。本次各版都实际重跑，没有挪用不同条件下的旧均值。', '',
      '冻结检验23个新布局×3尺度，每版69场；其中普通14布局×3尺度=42场，边界朝外9场，其余各6场。总源数、频道、接收朝向和半径均由共同生成器产生。秒/源先按每场T/N计算再平均，压力组分开报告；重复尺度不当作独立布局。','',
      '## 冻结后同场景平均时间','',
      '| 方案 | 普通42场 | 边界朝外9场 | 聚集6场 | 最小半径6场 | 边界内侧朝外6场 |',
      '|---|---:|---:|---:|---:|---:|']
    for v in ['fifth','eighth','boundary','reliable',selected]:
        lines.append('| '+LABELS[v]+' | '+' | '.join(f"{stats[m][v]['time_per_source_s']:.2f}" for m in ['mixed','edge_outward','rotated_cluster','minimum_range','edge_inset'])+' |')
    lines+=['','单位为秒/源。所有方案的69场均全部清除并完成发现证书检查。', '',
      '主线迭代 (3) 的完整分支没有另列重跑；本轮移植其巡检保护思想，并在开发组比较保留与取消保护。上表不应理解为已经检验了该对话的所有完整版本。', '',
      '普通组按14个独立布局作5000次配对自助重采样，节省为正表示整合版更快：', '',
      '| 对照 | 平均节省/秒每源 | 95%区间/秒每源 |', '|---|---:|---:|']
    for v in ['fifth','eighth','boundary','reliable']:
        c=comparisons['mixed'][v];lo,hi=c['layout_bootstrap_95ci']
        lines.append(f"| {LABELS[v]} | {c['mean_saved_s']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines+=['', '相对主线迭代 (2) 和自主启发式版的区间均包含0，当前样本不足以确认整合版对它们具有稳定优势。聚集组均值也略慢于自主启发式版。普通平均仍比400秒/源多40.65秒，目标尚未达到。', '',
      '## 普通场景分布','', '| 指标 | 原第八轮 | 整合版 |','|---|---:|---:|']
    for k,label in [('median','中位数'),('p90','90%分位数'),('maximum','最大值'),('distance_m','每场平均路程/米'),('measures','每场平均测量次数'),('clear_failures','每场平均失败清除次数'),('runtime_s','每场平均程序计算秒')]:
        lines.append(f"| {label} | {a[k]:.2f} | {b[k]:.2f} |")
    lines += [f"| ≤400秒/源 | {a['le400']}/{a['runs']} | {b['le400']}/{b['runs']} |",'',
      f"普通组相对第八轮：{p['faster']}场更快、{p['slower']}场更慢、{p['ties']}场相同。平均移动时间减少{(a['distance_m']-b['distance_m'])/5:.2f}秒/场；测量时间变化{5*(b['measures']-a['measures']):+.2f}秒/场，换频时间变化{b['switches']-a['switches']:+.2f}秒/场。",'',
      '时间账为路程÷5＋5×测量次数＋换频次数＋5×成功清除次数＋3×失败清除次数。程序计算时间单独报告，未加进模拟执行时间；实际部署若规划等待无法与行动重叠，还需加上这部分等待。最大值相对第八轮改善，但整合版619.80仍高于第五轮和主线迭代 (2) 的约603.54秒/源。', '',
      '## 兜底和补救','', '| 场景 | 原第八轮触发场数 | 整合版触发场数 | 原第八轮兜底尝试总数 | 整合版总数 |','|---|---:|---:|---:|---:|']
    for m in stats:
        x=stats[m]['eighth'];y=stats[m][selected]
        lines.append(f"| {MODES[m]} | {x['fallback_runs']}/{x['runs']} | {y['fallback_runs']}/{y['runs']} | {round(x['fallback_attempts']*x['runs'])} | {round(y['fallback_attempts']*y['runs'])} |")
    selected_rows=[r for r in held if r['variant']==selected]
    keys=set(k for r in selected_rows for k in r.get('integration_counts',{}))
    counts={k:sum(r.get('integration_counts',{}).get(k,0) for r in selected_rows) for k in keys}
    lines+=['',f"整合版实际开展{counts.get('boundary_episodes',0)}次边界补救，其中{counts.get('boundary_success',0)}次成功、{counts.get('boundary_failed',0)}次失败；接受{counts.get('guard_relaxations',0)}次巡检上限放宽，因最后发现风险拒绝{counts.get('discovery_risk_rejections',0)}次。边界试清除和外侧测量的成本已计入总时间，不能只报最后光学兜底减少而漏掉新增补救开销。",'',
      '## 开发消融，不作为独立验证','', '| 配置 | 普通秒/源 | 每场路程/米 | 每场测量次数 | 每场计算秒 |','|---|---:|---:|---:|---:|']
    for v in ['integrated','expanded','free','no_sensing']:
        x=dev['mixed'][v]
        lines.append(f"| {LABELS[v]} | {x['time_per_source_s']:.2f} | {x['distance_m']:.2f} | {x['measures']:.2f} | {x['runtime_s']:.2f} |")
    lines+=['', '这是完整策略的消融：关闭补测也会改变信息和后续路线，不能解释为每次补测都带来固定收益。开发在选参数前记录规则，冻结后未用验证结果重新调参。','',
      '## 仍存在的限制','',
      '- 未来服务出口、收到/收不到分支、失败恢复与总数为16的概率仍是有限假设下的估计。实际动作位置比只用中心更具体，但尚非完整反馈树的最优求解。',
      '- 巡检约束仍是参考路线保护。放宽和最后发现风险不是严格时间保证；未完成站点回访导致预算不可行时有显式修复计数，不隐藏这类例外。',
      '- 边界补救采用观测门槛及固定偏移，可能选错服务时机；保留失败停用和有限覆盖兜底。样本全部成功不能证明未来不触发兜底。',
      '- 普通组只有14个独立布局，压力组更少；均值、尾部及置信区间必须一起看。全部结果来自自建离线模拟，不能当作官方成绩。','',
      '普通组相对原第八轮的最大退步案例：','', '| 种子 | 尺度/米 | 原第八轮秒/源 | 整合版秒/源 | 整场增加/秒 |','|---|---:|---:|---:|---:|']
    for r in p['worst_regressions']:
        lines.append(f"| {r['seed']} | {r['scale']:.0f} | {r['before_s']:.2f} | {r['after_s']:.2f} | {r['extra_total_s']:.2f} |")
    lines+=['', '对这三场的已记录时间账作分解，不新增仿真，也不据此修改已冻结参数：', '',
      '- 1012005、100米尺度：整场多668.46秒，其中走路多656.46秒，测量反而少5秒。仍然值得优先修正目标服务顺序和出口预测，不能将退步归因于测量太多。该场有7次风险否决，但仅凭计数不能证明是约束导致绕路。',
      '- 1012010、900米尺度：16个源，实际到访站数由10增至19；整场多452.81秒，其中走路多159.81秒、测量多240秒、换频多44秒。提前结束巡检的机会仍未得到充分保护；该场没有放宽巡检约束，说明只在超过路程上限时审查发现风险并不充分。',
      '- 1012006、900米尺度：整场多336.02秒，其中走路多254.02秒、测量多65秒。路线连接和补测收益预测都有继续校准的空间。', '',
      '下一步优先研究：让上限以内的路线也更充分考虑最后发现时刻；改进服务出口及失败后的回访代价预测；只对多个候选收益接近或尾部风险较高的决策增加搜索预算。这些是本轮结果提出的待验证方向，不能算作已实现的提速。', '',
      '## 文件','', '- 运行入口：上一级run_integrated_q4.py。','- 逐场对照：results/heldout/comparison.csv。',
            '- 配对统计、置信区间与审计：results/final_audit.json。','- 冻结参数与源代码：results/frozen_config.json、results/frozen_source。','']
    (ROOT/'整合思路与有限仿真统计.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k not in ['stages','comparisons']},ordinary=stats['mixed'],comparisons=comparisons['mixed']),indent=2))


if __name__=='__main__':main()
