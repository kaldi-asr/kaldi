# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2016    Vijayaditya Peddinti
# Apache 2.0.


""" This module contains the implementation of the TDNN layer.
"""

import libs.nnet3.xconfig.utils as xutils
from libs.nnet3.xconfig.basic_layers import XconfigBasicLayer
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

class XconfigTdnnLayer(XconfigBasicLayer):
    """This class is for parsing lines like
    tdnn-relu-renorm-layer name=tdnn1 dim=1024 splice-indexes=-3,0,3 subset-dim=512

    It is similar to XconfigBasicLayer except for the way in which the input
    splicing is done. So we derive this class from XconfigBasicLayer.
    """

    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token in [ 'tdnn-relu-layer', 'tdnn-relu-renorm-layer',
                                'tdnn-sigmoid-layer', 'tdnn-tanh-layer' ]
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)


    def set_default_configs(self):

        super(XconfigTdnnLayer, self).set_default_configs()

        self.config['splice-indexes'] = ''
        self.config['subset-dim'] = -1

    def check_configs(self):

        if self.config['splice-indexes'] == '':
            raise RuntimeError("splice-indexes must be non-empty")
        super(XconfigTdnnLayer, self).check_configs()


    def _generate_config(self):
        split_layer_name = self.layer_type.split('-')
        assert split_layer_name[-1] == 'layer'
        # ignore the first 'tdnn' and the last 'layer'
        nonlinearities = split_layer_name[1:-1]

        # by 'descriptor_final_string' we mean a string that can appear in
        # config-files, i.e. it contains the 'final' names of nodes.
        input_desc = self.descriptors['input']['final-string']
        input_dim = self.descriptors['input']['dim']
        splice_indexes = self.get_splice_indexes()
        input_desc, input_dim, sp_configs = self.splice_input(input_desc,
                input_dim, splice_indexes, self.config['subset-dim'],
                '{0}.input-subset'.format(self.name))

        return sp_configs + self._add_components(input_desc, input_dim, nonlinearities)

    def get_splice_indexes(self):
        try:
            return map(lambda x: int(x), self.config['splice-indexes'].split(","))
        except ValueError:
            raise RuntimeError("Invalid value for splice-indexes.")

    @staticmethod
    def splice_input(input_desc, input_dim,
                     splice_indexes, subset_dim = -1,
                     dim_range_node_name = None ):
        """Convenience function to create an appended descriptor with the
        splice_indexes.
        """

        configs = []
        try:
            zero_index = splice_indexes.index(0)
        except ValueError:
            zero_index = None

        if subset_dim > 0:
            assert(dim_range_node_name is not None)
            # if subset_dim is specified the script expects a zero
            # in the splice indexes
            assert(zero_index is not None)
            line = ("dim-range-node name={0}"
                    " input-node={1}"
                    " dim-offset={2}"
                    " dim={3}"
                    "".format(dim_range_node_name,
                              input_desc, 0, subset_dim))
            configs.append(line)
            subset_desc = dim_range_node_name

        else:
            subset_desc = input_desc
            subset_dim = input_dim

        appended_descriptors = []
        appended_dimension = 0
        for j in range(len(splice_indexes)):
            if j == zero_index:
                appended_descriptors.append(input_desc)
                appended_dimension += input_dim
                continue
            appended_descriptors.append('Offset({0}, {1})'.format(subset_desc, splice_indexes[j]))
            appended_dimension += subset_dim
        return ["Append({0})".format(", ".join(appended_descriptors)),
                appended_dimension,
                configs]
