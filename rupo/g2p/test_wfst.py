import unittest

from rupo.g2p.wfst import WeightedFiniteStateTransducer, LogWeight


class TestPatterns(unittest.TestCase):
    def test_fst(self):
        fst = WeightedFiniteStateTransducer()
        fst.add_state()
        fst.add_state()
        fst.add_state()
        fst.add_state()
        fst.add_arc(0, 1, 'a', 'A', weight=LogWeight(9.0))
        fst.add_arc(1, 2, 'c', 'C', weight=LogWeight(5.0))
        fst.add_arc(0, 3, 'b', 'B', weight=LogWeight(6.0))
        fst.add_arc(3, 2, 'd', 'D', weight=LogWeight(7.0))
        fst.set_final(2)
        print(fst)
        print(fst.get_shortest_path())
        print(fst.get_reversal())
        print(fst.get_shortest_path(mode="Tropical"))
        print(LogWeight.one() + LogWeight.one())
        print(LogWeight.zero() + LogWeight(5.0))
