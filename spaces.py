from collections import OrderedDict

primitives_1 = OrderedDict([('primitives_normal', [['skip_connect',
                                                    'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5']]),
                             ('primitives_reduct', [['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5']])])

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

primitives_2 = OrderedDict([('primitives_normal', 14 * [PRIMITIVES]),
                            ('primitives_reduct', 14 * [PRIMITIVES])])

PRIMITIVES_SMALL = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_3x1',
    'sep_conv_1x3',
    'dil_conv_3x3',
    'dil_conv_3x1',
    'dil_conv_1x3',
]

primitives_3 = OrderedDict([('primitives_normal', 14 * [PRIMITIVES_SMALL]),
                            ('primitives_reduct', 14 * [PRIMITIVES_SMALL])])

spaces_dict = {
    's1': primitives_1, # space from https://openreview.net/forum?id=H1gDNyrKDS
    's2': primitives_2, # original DARTS space
    's3': primitives_3, # space with 1D conv
}