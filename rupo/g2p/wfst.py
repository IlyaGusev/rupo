import sys
from numpy import log, exp


class LogWeight:
    def __init__(self, w: float):
        self.weight = w

    @staticmethod
    def zero() -> 'LogWeight':
        return LogWeight(float('inf'))

    @staticmethod
    def one() -> 'LogWeight':
        return LogWeight(0.0)

    @staticmethod
    def prod(x: 'LogWeight', y: 'LogWeight') -> 'LogWeight':
        return LogWeight(x.weight + y.weight)

    @staticmethod
    def div(x: 'LogWeight', y: 'LogWeight') -> 'LogWeight':
        return LogWeight(x.weight - y.weight)

    def __lt__(self, other):
        return self.weight > other.weight

    def __eq__(self, other):
        return self.weight == other.weight

    def __ne__(self, other):
        return self.weight != other.weight

    def __gt__(self, other):
        return self.weight < other.weight

    def __ge__(self, other):
        return self.weight <= other.weight

    def __add__(self, other):
        if self == LogWeight.zero() and other == LogWeight.zero():
            return LogWeight.zero()
        if self == LogWeight.zero():
            return other
        if other == LogWeight.zero():
            return self
        return LogWeight(-log(exp(-self.weight) + exp(-other.weight)))

    def __mul__(self, other):
        return LogWeight(self.weight + other.weight)

    def __floordiv__(self, other):
        if self == LogWeight.zero() or other == LogWeight.zero():
            return LogWeight.zero()
        return LogWeight(self.weight - other.weight)

    def __str__(self):
        return "Log -> " + str(self.weight)

    def __repr__(self):
        return self.__str__()


class State:
    def __init__(self, state_id: int, weight: LogWeight=LogWeight.zero(), is_final: bool=False):
        self.id = state_id
        self.weight = weight  # type: LogWeight
        self.is_final = is_final
        self.arcs = []

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return self.__str__()


class Arc:
    def __init__(self, next_state_id: int, input_label: str, output_label: str, weight: LogWeight=LogWeight.zero()):
        self.input_label = input_label
        self.output_label = output_label
        self.weight = weight
        self.next_state_id = next_state_id

    def __str__(self):
        return '-> {} / {}:{} / {}'.format(self.next_state_id, self.input_label, self.output_label, self.weight)

    def __repr__(self):
        return self.__str__()


class WeightedFiniteStateTransducer:
    EPSILON = "eps"

    def __init__(self):
        self.states = []

    def add_state(self):
        self.states.append(State(len(self.states)))

    def add_arc(self, src_state_id: int, dst_state_id: int, input_label: str,
                output_label: str=EPSILON, weight: LogWeight=LogWeight.zero()):
        self.states[src_state_id].arcs.append(Arc(dst_state_id, input_label, output_label, weight))

    def set_final(self, state_id: int, weight: LogWeight=LogWeight.zero()):
        self.states[state_id].is_final = True
        self.states[state_id].weight = weight

    def get_reversal(self):
        new_transducer = WeightedFiniteStateTransducer()
        new_transducer.states = [State(0)] + [State(i+1) for i in range(len(self.states))]
        for state in self.states:
            for arc in state.arcs:
                new_transducer.add_arc(arc.next_state_id+1, state.id+1, arc.input_label, arc.output_label, arc.weight)
            if state.is_final:
                new_transducer.add_arc(0, state.id+1, WeightedFiniteStateTransducer.EPSILON, weight=LogWeight.zero())
            new_transducer.set_final(1)
        return new_transducer

    def get_shortest_path(self, mode="Log"):
        distances = [None for _ in range(len(self.states))]
        prev_graph = [None for _ in range(len(self.states))]
        distances[0] = LogWeight.zero()
        border = set()
        current_id = 0
        value = LogWeight.zero()
        while True:
            for arc in self.states[current_id].arcs:
                if mode == "Log":
                    border.add(arc.next_state_id)
                    if distances[arc.next_state_id] is None:
                        distances[arc.next_state_id] = LogWeight.zero()
                    distances[arc.next_state_id] += value + arc.weight
                if mode == "Tropical":
                    if distances[arc.next_state_id] is None or value + arc.weight < distances[arc.next_state_id]:
                        border.add(arc.next_state_id)
                        distances[arc.next_state_id] = value + arc.weight
                        prev_graph[arc.next_state_id] = current_id
            if len(border) == 0:
                break
            current_id = list(border)[0]
            value = distances[current_id]
            for i in border:
                if mode == "Log":
                    value = distances[i]
                    current_id = i
                elif mode == "Tropical":
                    if distances[i] < value:
                        value = distances[i]
                        current_id = i
            border.remove(current_id)
        final_id = None
        for state in self.states:
            if state.is_final:
                final_id = state.id
        answer = []
        if mode == "Tropical":
            current_id = final_id
            while prev_graph[current_id] is not None:
                answer = [current_id] + answer
                current_id = prev_graph[current_id]
            answer = [current_id] + answer
        return answer, distances

    def paths(self):
        return [path[::-1] for path in self.__find_paths(self.states[0])]

    def __find_paths(self, state):
        answers = []
        for arc in state.arcs:
            next_id = arc.next_state_id
            next_state = self.states[next_id]
            if next_state.is_final:
                answers.append([arc])
            else:
                for answer in self.__find_paths(next_state):
                    if len(answer) != 0:
                        answers.append(answer + [arc])
        return answers

    def __str__(self):
        result = ""
        for state in self.states:
            for arc in state.arcs:
                result += '{} {}\n'.format(state.id, arc)
        return result
