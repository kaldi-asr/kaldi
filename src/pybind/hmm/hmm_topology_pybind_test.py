#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import shutil
import unittest

from tempfile import mkdtemp

import kaldi


class TestHmmTopology(unittest.TestCase):

    def test(self):
        tmp = mkdtemp()
        rxfilename = '{}/topo'.format(tmp)
        # build a sample topology, which is copied from
        #https://github.com/kaldi-asr/kaldi/blob/master/src/hmm/hmm-topology.h#L46
        topo_str = '''
<Topology>
<TopologyEntry>
<ForPhones>
1 2 3 4 5 6 7 8
</ForPhones>
<State> 0 <PdfClass> 0 <Transition> 0 0.5 <Transition> 1 0.5 </State>
<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>
<State> 2 <PdfClass> 2 <Transition> 2 0.5 <Transition> 3 0.5 </State>
<State> 3 </State>
</TopologyEntry>
</Topology>
        '''

        with open(rxfilename, 'w') as f:
            f.write(topo_str)

        ki = kaldi.Input()
        is_opened, = ki.Open(rxfilename, read_header=False)
        self.assertTrue(is_opened)

        topo = kaldi.HmmTopology()
        topo.Read(ki.Stream(), binary=False)
        ki.Close()

        topo.Check()
        self.assertTrue(topo.IsHmm())
        self.assertEqual(topo.GetPhones(), [1, 2, 3, 4, 5, 6, 7, 8])
        entry = topo.TopologyForPhone(1)
        self.assertEqual(len(entry), 4)  # 4 states: 0--3
        for i in range(3):
            self.assertEqual(entry[i].forward_pdf_class, i)
            self.assertEqual(entry[i].self_loop_pdf_class, i)

        # -1 is kNoPdf
        self.assertEqual(entry[3].forward_pdf_class, -1)
        self.assertEqual(entry[3].self_loop_pdf_class, -1)

        shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
