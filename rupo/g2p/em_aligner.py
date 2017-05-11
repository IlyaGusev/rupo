from rupo.g2p.wfst import WeightedFiniteStateTransducer, LogWeight
from collections import defaultdict
from rupo.settings import RU_G2P_DICT_PATH


class ExpectationMaximizationFST:
    @staticmethod
    def build_fsas(g, p, max_x: int=2, max_y: int=2, del_x: bool=False, del_y: bool=False):
        fsas = []
        for i in range(len(g)):
            graphemes = g[i]
            phonemes = p[i]
            fsa = ExpectationMaximizationFST.build_fsa(graphemes, phonemes, max_x, max_y, del_x, del_y)
            fsas.append(fsa)
        return fsas

    @staticmethod
    def build_fsa(graphemes, phonemes, max_x: int=2, max_y: int=2, del_x: bool=False, del_y: bool=False):
        print(graphemes, phonemes)
        fsa = WeightedFiniteStateTransducer()
        visited = [[False for j in range(len(phonemes)+1)] for i in range(len(graphemes)+1)]
        for i in range(len(graphemes) + 1):
            for j in range(len(phonemes) + 1):
                fsa.add_state()
        queue = {0}
        while len(queue) != 0:
            current_id = list(queue)[0]
            queue.remove(current_id)
            row = current_id // (len(phonemes) + 1)
            col = current_id % (len(phonemes) + 1)
            visited[row][col] = True
            for i in range(1-int(del_x), max_x+1):
                for j in range(1 - int(del_y), max_y+1):
                    if i == j == 2:
                        continue
                    new_row = row + i
                    new_col = col + j
                    if new_col > len(phonemes) or new_row > len(graphemes):
                        continue
                    new_id = new_row * (len(phonemes) + 1) + new_col
                    input_label = graphemes[row:new_row]
                    output_label = phonemes[col:new_col]
                    input_label = input_label if input_label != "" else " "
                    output_label = output_label if input_label != "" else " "
                    fsa.add_arc(current_id, new_id, input_label, output_label, weight=LogWeight(10.0))
                    if not visited[new_row][new_col]:
                        queue.add(new_id)
        fsa.set_final((len(phonemes) + 1)*(len(graphemes) + 1)-1)

        if not (del_x and del_y):
            bad_states = set()
            for diff in range(0, max(len(graphemes), len(phonemes)+1)):
                i = len(graphemes) - diff
                if i >= 0:
                    for j in range(len(phonemes) + 1):
                        fsa, bad_states = ExpectationMaximizationFST.__prune_state(fsa, i, j, len(phonemes), bad_states)
                j = len(phonemes) - diff
                if j >= 0:
                    for i in range(len(graphemes) + 1):
                        fsa, bad_states = ExpectationMaximizationFST.__prune_state(fsa, i, j, len(phonemes), bad_states)
            new_states = []
            mapping = {}
            for state in fsa.states:
                if state.id not in bad_states:
                    new_states.append(state)
                    mapping[state.id] = len(new_states)-1
                    new_states[-1].id = len(new_states)-1
            fsa.states = new_states
            for state in fsa.states:
                for arc in state.arcs:
                    arc.next_state_id = mapping[arc.next_state_id]
        return fsa

    @staticmethod
    def __prune_state(fsa, i, j, col_len, bad_states):
        current_id = i * (col_len + 1) + j
        state = fsa.states[current_id]
        new_arcs = []
        for arc in state.arcs:
            if arc.next_state_id not in bad_states:
                new_arcs.append(arc)
        fsa.states[current_id].arcs = new_arcs
        if len(state.arcs) == 0 and not state.is_final:
            bad_states.add(current_id)
        return fsa, bad_states

    @staticmethod
    def expectation_step(fsas, gamma, total):
        for i, fsa in enumerate(fsas):
            path_a, alpha = fsa.get_shortest_path(mode="Log")
            path_b, beta = fsa.get_reversal().get_shortest_path(mode="Log")
            if i == 0:
                print(fsa)
                print(path_a, alpha)
                print(path_b, beta)
            for state in fsa.states:
                for arc in state.arcs:
                    if gamma.get(arc.input_label) is None:
                        gamma[arc.input_label] = {}
                    if gamma[arc.input_label].get(arc.output_label) is None:
                        gamma[arc.input_label][arc.output_label] = LogWeight.zero()

                    val = (alpha[state.id] * arc.weight * beta[arc.next_state_id + 1]) // beta[1]
                    # if len(arc.input_label) == 2 or len(arc.output_label) == 2:
                    #     val = val // (LogWeight.one() + LogWeight.one() + LogWeight.one())
                    gamma[arc.input_label][arc.output_label] = gamma[arc.input_label][arc.output_label] + val
                    total = total + val
        return gamma, total

    @staticmethod
    def maximization_step(fsas, gamma, total):
        for g, value in gamma.items():
            for p, value2 in value.items():
                gamma[g][p] = value2 // total
        for i, fsa in enumerate(fsas):
            for j, state in enumerate(fsa.states):
                for k, arc in enumerate(state.arcs):
                    fsas[i].states[j].arcs[k].weight = gamma[arc.input_label][arc.output_label]
        total = LogWeight.zero()
        for g, value in gamma.items():
            for p, value2 in value.items():
                gamma[g][p] = LogWeight.zero()
        return gamma, total

    @staticmethod
    def run():
        with open(RU_G2P_DICT_PATH, 'r', encoding='utf-8') as r:
            lines = r.readlines()[::20]
            g = [line.strip().split("\t")[0] for line in lines]
            p = [line.strip().split("\t")[1] for line in lines]
            fsas = ExpectationMaximizationFST.build_fsas(g, p, del_x=False, del_y=False)
            gamma = {}
            total = LogWeight.zero()
            for i in range(5):
                gamma, total = ExpectationMaximizationFST.expectation_step(fsas, gamma, total)
                gamma, total = ExpectationMaximizationFST.maximization_step(fsas, gamma, total)
                for fsa in fsas[::20]:
                    path = fsa.get_shortest_path(mode="Tropical")[0]
                    i = 0
                    g = []
                    p = []
                    while i != len(path):
                        state = fsa.states[path[i]]
                        i += 1
                        for arc in state.arcs:
                            if arc.next_state_id == path[i]:
                                g.append(arc.input_label)
                                p.append(arc.output_label)
                    print(g, p)

ExpectationMaximizationFST.run()