"""Small bounded metaheuristics for a changing open-route snapshot.

Implemented independently from the cited papers: randomized VND, ILS,
ALNS with regret repair and annealing, and a width-limited rollout beam.
Only surrogate snapshot cost is optimized; no online-optimality claim.
"""
import math
import random


class RouteModel:
    def __init__(self, labels, points, start, task_radii, effects, blocked,
                 uncertainty_weight=.35, unknown_channels=0, discovery=None):
        self.labels = list(labels)
        self.n = len(labels)
        self.distance = [[math.dist(a, b)/5 for b in points] for a in points]
        self.from_start = [math.dist(start, p)/5 for p in points]
        self.task_radii = dict(task_radii)
        self.effects, self.blocked = effects, set(blocked)
        self.weight, self.unknown = uncertainty_weight, unknown_channels
        self.stations = {i for i, k in enumerate(labels) if k[0] == 'station'}
        self.calls = 0
        self.discovery = discovery

    def cost(self, route):
        self.calls += 1
        if not route:
            return 0.
        score = self.from_start[route[0]]
        score += sum(self.distance[a][b] for a,b in zip(route,route[1:]))
        discovery_costs = {}
        if self.discovery and self.unknown > 4:
            masks, hidden = self.discovery
            total = hidden.bit_count()
            found = 20-self.unknown
            mean_remaining = (max(10,found)+16)/2-found
            probability16 = 1/(17-max(10,found))
            last = None
            for node in route:
                if node in self.stations:
                    seen = 1-hidden.bit_count()/total if total else 0.
                    continuing = 1-probability16*seen**(16-found)
                    edge = self.from_start[node] if last is None else self.distance[last][node]
                    # A soft expected saving on optional future station travel;
                    # tasks themselves are mandatory even if discovery stops.
                    score -= (1-continuing)*edge
                    discovery_costs[node] = continuing*6*(self.unknown-mean_remaining*seen)
                    hidden &= ~masks[node]
                last = node
        if not self.weight:
            # The precedence constraint also applies in geometry-only ablations.
            last_station = max((j for j,k in enumerate(route) if k in self.stations), default=-1)
            return score+1e7*sum(j < last_station and k in self.blocked for j,k in enumerate(route))
        radii = dict(self.task_radii)
        remaining_stations = len(self.stations)
        for node in route:
            if node in self.stations:
                remaining_stations -= 1
                score += discovery_costs.get(node,6*self.unknown)
                for task, predicted, possible in self.effects.get(node, []):
                    if task in radii and possible and radii[task] > 19.8:
                        score += 6.
                        radii[task] = min(radii[task], predicted)
            else:
                if node in self.blocked and remaining_stations:
                    score += 1e7
                radius = radii.pop(node, 0.)
                score += 5+self.weight*max(0., radius-19.8)+(6 if radius > 19.8 else 0)
        return score

    def allowed_position(self, node, order, index):
        if node in self.blocked:
            return not any(k in self.stations for k in order[index:])
        if node in self.stations:
            return not any(k in self.blocked for k in order[:index])
        return True

    def insertion_options(self, node, order):
        options = []
        for i in range(len(order)+1):
            if not self.allowed_position(node, order, i):
                continue
            extra = self.from_start[node] if not i else self.distance[order[i-1]][node]
            if i < len(order):
                extra += self.distance[node][order[i]]
                extra -= self.from_start[order[i]] if not i else self.distance[order[i-1]][order[i]]
            options.append((extra, i))
        return sorted(options)

    def repair(self, partial, removed, regret=1):
        order, pending = list(partial), list(removed)
        while pending:
            choices = []
            for k in pending:
                options = self.insertion_options(k, order)
                best, index = options[0]
                gap = sum(v-best for v,_ in options[1:regret])
                choices.append((-gap, best, k, index))
            _, _, k, index = min(choices)
            order.insert(index, k)
            pending.remove(k)
        return order

    def nearest_completion(self, prefix, remaining):
        order, rest = list(prefix), set(remaining)
        while rest:
            possible = [k for k in rest if k not in self.blocked or not (rest & self.stations)]
            k = min(possible, key=lambda k: (self.distance[order[-1]][k] if order else self.from_start[k], k))
            order.append(k)
            rest.remove(k)
        return order


def neighbour(order, rng, kind):
    n = len(order)
    if n < 2:
        return list(order)
    i,j = sorted(rng.sample(range(n), 2))
    result = list(order)
    if kind == 0:  # 2-opt
        result[i:j+1] = reversed(result[i:j+1])
    elif kind == 1:  # one-point relocation
        result.insert(j, result.pop(i))
    elif kind == 2:  # swap
        result[i],result[j] = result[j],result[i]
    else:  # Or-opt: relocate a block of two or three nodes
        size = min(rng.choice((2,3)), n-i)
        block = result[i:i+size]
        del result[i:i+size]
        j = rng.randrange(len(result)+1)
        result[j:j] = block
    return result


def descent(model, order, rng, rounds=4, probes=16):
    current = list(order)
    score = model.cost(current)
    for _ in range(rounds):
        changed = False
        for kind in range(4):
            for _ in range(probes):
                candidate = neighbour(current, rng, kind)
                value = model.cost(candidate)
                if value < score-1e-7:
                    current,score,changed = candidate,value,True
        if not changed:
            break
    return current


def optimize(model, initial, method='alns', iterations=48, seed=8137, warm=None,
             beam_depth=3, beam_width=4, beam_branch=6):
    rng = random.Random(seed)
    candidates = [list(initial), model.repair([], range(model.n), 2),
                  model.nearest_completion([], range(model.n))]
    if warm is not None:
        # Source positions change after observations: re-evaluate, never reuse old costs.
        remaining = set(range(model.n))-set(warm)
        candidates.append(model.repair(warm, remaining, 2))
    current = min(candidates, key=model.cost)
    current = descent(model, current, rng, rounds=2, probes=12)
    best, best_cost = list(current), model.cost(current)
    current_cost = best_cost
    stats = {'method':method, 'iterations':0, 'accepted_worse':0,
             'initial_cost':model.cost(initial), 'operator_uses':[0]*4}
    if method == 'vnd':
        best = descent(model, current, rng, rounds=6, probes=24)
    elif method == 'ils':
        for k in range(iterations//4):
            if model.n < 4:
                break
            cuts = sorted(rng.sample(range(1, model.n), min(3, model.n-1)))
            chunks = [current[a:b] for a,b in zip([0]+cuts, cuts+[model.n])]
            candidate = chunks[0]+chunks[2]+chunks[1]+chunks[3]
            candidate = descent(model, candidate, rng, rounds=2, probes=10)
            value = model.cost(candidate)
            if value < best_cost:
                best,best_cost = list(candidate),value
            current = candidate if value < current_cost or k % 4 == 3 else list(best)
            current_cost = model.cost(current)
            stats['iterations'] += 1
    elif method == 'beam':
        beam = [([], set(range(model.n)))]
        for depth in range(min(beam_depth, model.n)):
            expanded = []
            for prefix, remaining in beam:
                admissible = [k for k in remaining if k not in model.blocked or not (remaining & model.stations)]
                nearest = sorted(admissible, key=lambda k: model.distance[prefix[-1]][k] if prefix else model.from_start[k])[:beam_branch]
                candidates_first = list(dict.fromkeys(nearest+[k for k in best if k in admissible][:2]))
                for k in candidates_first:
                    new_prefix, rest = prefix+[k], remaining-{k}
                    completion = model.nearest_completion(new_prefix, rest)
                    value = model.cost(completion)
                    expanded.append((value,new_prefix,rest,completion))
                    if value < best_cost:
                        best,best_cost = completion,value
            expanded.sort(key=lambda v:(v[0],v[1]))
            beam = [(prefix,remaining) for _,prefix,remaining,_ in expanded[:beam_width]]
            stats['iterations'] += 1
    elif method == 'alns':
        weights,repair_weights = [1.]*4,[1.]*3
        for k in range(iterations):
            if model.n < 3:
                break
            op = rng.choices(range(4), weights=weights)[0]
            regret = rng.choices((1,2,3), weights=repair_weights)[0]
            q = rng.randint(2, min(7, model.n))
            stats['operator_uses'][op] += 1
            if op == 0:
                removed = rng.sample(current, q)
            elif op == 1:
                pivot = rng.choice(current)
                removed = sorted(current, key=lambda x:model.distance[pivot][x])[:q]
            elif op == 2:
                def saving(j):
                    x = current[j]
                    value = model.from_start[x] if not j else model.distance[current[j-1]][x]
                    if j+1 < len(current):
                        value += model.distance[x][current[j+1]]
                        value -= model.from_start[current[j+1]] if not j else model.distance[current[j-1]][current[j+1]]
                    return value
                removed = [current[j] for j in sorted(range(model.n),key=saving,reverse=True)[:q]]
            else:
                begin = rng.randrange(model.n-q+1)
                removed = current[begin:begin+q]
            partial = [x for x in current if x not in removed]
            candidate = model.repair(partial, removed, regret)
            # Allow non-geometric improvements, e.g. waiting for a useful station.
            candidate = descent(model, candidate, rng, rounds=1, probes=3)
            value = model.cost(candidate)
            reward = .2
            if value < best_cost-1e-7:
                best,best_cost,reward = list(candidate),value,8.
            temperature = max(1., .006*stats['initial_cost'])*(0.95**k)
            if value < current_cost or rng.random() < math.exp(min(0., (current_cost-value)/temperature)):
                improved_current = value < current_cost
                if value > current_cost:
                    stats['accepted_worse'] += 1
                current,current_cost = candidate,value
                reward = max(reward, 3. if improved_current else 1.)
            weights[op] = .85*weights[op]+.15*reward
            repair_weights[regret-1] = .85*repair_weights[regret-1]+.15*reward
            stats['iterations'] += 1
        stats.update(destroy_weights=weights,repair_weights=repair_weights)
    else:
        raise ValueError('Unknown heuristic')
    final_cost = model.cost(best)
    if final_cost > stats['initial_cost']+1e-7:
        best,final_cost = list(initial),stats['initial_cost']
    assert sorted(best) == list(range(model.n))
    stats.update(final_cost=final_cost, cost_evaluations=model.calls)
    return best,stats
