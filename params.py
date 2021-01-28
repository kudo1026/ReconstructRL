# params file contain parameters that comes with the environment,
# and are usually not supposed to be changed

import numpy as np
from numpy import array

n_nbs = 5
net_name = "rnn_vgg16unet3_gruunet4.64.3"
net_path = "/home/kudo/devs/FreeViewSynthesis/exp/experiments/tat_nbs5_s0.25_p192_rnn_vgg16unet3_gruunet4.64.3/net_0000000000749999.params"
pw_dir = "/home/kudo/external_disk/FVS_data/ibr3d_tat/training/Truck/dense/ibr3d_pw_0.50"

N_DISCRETE_ACTIONS = 36
TOTAL_AZIMUTH_ANGLE = 360

action_map = [(array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.99691859,  0.00879406,  0.07794859],
         [-0.00829108,  0.99994268, -0.00677394],
         [-0.0780037 ,  0.00610678,  0.99693437]]),
  array([-1.15893849, -0.45901755,  3.91884407])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.99459148, -0.00984229, -0.10339693],
         [ 0.01107075,  0.99987471,  0.01131387],
         [ 0.10327262, -0.01239736,  0.99457583]]),
  array([-1.14005212, -0.41290245,  3.46211008])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.96338146, -0.02469214, -0.26699524],
         [ 0.03259117,  0.99915119,  0.02519351],
         [ 0.26614653, -0.03297265,  0.96336848]]),
  array([-0.97006547, -0.34773682,  3.12503621])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.91076774, -0.02763621, -0.41199316],
         [ 0.04393567,  0.99857955,  0.03014184],
         [ 0.41057494, -0.04555341,  0.91068826]]),
  array([-0.70796913, -0.28382099,  2.90557446])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.84303165, -0.01522974, -0.5376483 ],
         [ 0.0345407 ,  0.99906868,  0.02585956],
         [ 0.53675374, -0.04037118,  0.84277256]]),
  array([-0.38958232, -0.24954685,  2.79217653])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.76156324,  0.00970139, -0.64801799],
         [ 0.00179068,  0.99985264,  0.0170731 ],
         [ 0.64808813, -0.01416263,  0.76143365]]),
  array([-0.04993173, -0.26796758,  2.76616921])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.66113547,  0.04094184, -0.74914862],
         [-0.04954571,  0.99871286,  0.01085595],
         [ 0.74862882,  0.02993984,  0.66231299]]),
  array([ 0.27120845, -0.34716863,  2.80808861])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.53291802,  0.07251831, -0.84305366],
         [-0.11299341,  0.99349664,  0.01403288],
         [ 0.83858862,  0.08778113,  0.53764635]]),
  array([ 0.53425041, -0.48540882,  2.90025058])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.36695465,  0.09786743, -0.92507635],
         [-0.17853963,  0.98337202,  0.03321255],
         [ 0.91294463,  0.15297529,  0.37832615]]),
  array([ 0.69890856, -0.66389603,  3.02556101])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.15842072,  0.11076666, -0.98113894],
         [-0.23153534,  0.97014798,  0.0721407 ],
         [ 0.95984075,  0.21573975,  0.17933794]]),
  array([ 0.73567618, -0.84342851,  3.1673305 ])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.08452382,  0.10882146, -0.99046131],
         [-0.25718801,  0.95795358,  0.12719772],
         [ 0.96265781,  0.26548601, -0.05298232]]),
  array([ 0.64580844, -0.98083963,  3.31512838])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.33685705,  0.09498144, -0.93675282],
         [-0.25201065,  0.94950508,  0.18689767],
         [ 0.90720337,  0.29902949, -0.29591115]]),
  array([ 0.45929268, -1.06753495,  3.47556642])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.57233748,  0.07357453, -0.81671084],
         [-0.22091659,  0.94530856,  0.23997414],
         [ 0.78969973,  0.31777117, -0.52478169]]),
  array([ 0.19588275, -1.11072531,  3.64594279])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.76883313,  0.04838706, -0.63761612],
         [-0.17093809,  0.94529283,  0.27785183],
         [ 0.61617838,  0.32261457, -0.71850125]]),
  array([-0.12656211, -1.10761313,  3.799703  ])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.90765656,  0.02271783, -0.4190984 ],
         [-0.11219568,  0.94906383,  0.29443161],
         [ 0.40443998,  0.31426381, -0.85887517]]),
  array([-0.45227434, -1.04708318,  3.88841553])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-9.78848072e-01, -2.28231376e-04, -2.04588367e-01],
         [-6.05687796e-02,  9.55494589e-01,  2.88723940e-01],
         [ 1.95417181e-01,  2.95008540e-01, -9.35297860e-01]]),
  array([-0.64027164, -0.92362463,  3.85666623])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.99936336, -0.01861772, -0.03043438],
         [-0.02615512,  0.9624879 ,  0.27006102],
         [ 0.02426481,  0.2706851 , -0.96236209]]),
  array([-0.60229489, -0.7710766 ,  3.73924847])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.99356983, -0.03373079,  0.10807968],
         [-0.00589871,  0.96871597,  0.24810193],
         [-0.11306719,  0.24586906, -0.96268594]]),
  array([-0.39928395, -0.6248062 ,  3.61370113])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.97257535, -0.04747   ,  0.22769231],
         [ 0.00560466,  0.97388369,  0.22697828],
         [-0.23252049,  0.22202962, -0.94690922]]),
  array([-0.11365087, -0.49714764,  3.52714914])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.93749189, -0.06113431,  0.34259531],
         [ 0.01224756,  0.97804335,  0.20804137],
         [-0.34779153,  0.19923305, -0.91615896]]),
  array([ 0.18720641, -0.39139128,  3.51409387])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.8837111 , -0.0750776 ,  0.46197191],
         [ 0.01673936,  0.98134882,  0.19150531],
         [-0.46773335,  0.17696848, -0.8659721 ]]),
  array([ 0.44398871, -0.30805935,  3.6029558 ])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.80551785, -0.08764663,  0.5860538 ],
         [ 0.02132745,  0.98407208,  0.17648592],
         [-0.59218758,  0.1546616 , -0.79081835]]),
  array([ 0.618584  , -0.24384178,  3.7916101 ])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.69632234, -0.09639141,  0.71122703],
         [ 0.02823391,  0.98649489,  0.16134023],
         [-0.71717365,  0.13242552, -0.68419693]]),
  array([ 0.67523976, -0.19965124,  4.06140139])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.54899096, -0.09832408,  0.83002488],
         [ 0.03962038,  0.98887905,  0.14334733],
         [-0.83488871,  0.11158229, -0.53899002]]),
  array([ 0.57930313, -0.18207932,  4.37619046])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.36021393, -0.09039216,  0.92848004],
         [ 0.05677717,  0.99132495,  0.11853775],
         [-0.93114031,  0.09541542, -0.35195685]]),
  array([ 0.31601622, -0.20157896,  4.67447325])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[-0.13927448, -0.07157267,  0.9876639 ],
         [ 0.07829744,  0.99346607,  0.08303418],
         [-0.98715355,  0.0888961 , -0.13276052]]),
  array([-0.07462877, -0.26578369,  4.88267482])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.09143854, -0.04528521,  0.9947805 ],
         [ 0.09922385,  0.99440835,  0.03614778],
         [-0.99085499,  0.09540065,  0.09542062]]),
  array([-0.50197567, -0.37168819,  4.96921594])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.31045682, -0.01898983,  0.95039778],
         [ 0.11268996,  0.99348544, -0.01696053],
         [-0.94388428,  0.1123658 ,  0.3105743 ]]),
  array([-0.90800042, -0.503763  ,  4.97368585])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.50579038,  0.0022382 ,  0.86265351],
         [ 0.11522199,  0.99086124, -0.07012772],
         [-0.85492689,  0.13486657,  0.50091019]]),
  array([-1.28040537, -0.64750293,  4.93616132])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.67035915,  0.016249  ,  0.74185887],
         [ 0.1064368 ,  0.9873163 , -0.11780379],
         [-0.73436355,  0.15793194,  0.66012702]]),
  array([-1.61582079, -0.79018704,  4.87721015])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.80076323,  0.02274705,  0.59854893],
         [ 0.08772728,  0.98404901, -0.1547626 ],
         [-0.59252188,  0.17643727,  0.78599473]]),
  array([-1.90515721, -0.9169906 ,  4.80607634])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.89475125,  0.02269715,  0.44598771],
         [ 0.06213659,  0.98266454, -0.17466955],
         [-0.44222081,  0.18399796,  0.87783   ]]),
  array([-2.11637102, -1.00383262,  4.727974  ])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.95296158,  0.0183853 ,  0.302533  ],
         [ 0.03475615,  0.98494531, -0.16933618],
         [-0.30109175,  0.17188575,  0.93797604]]),
  array([-2.18011208, -1.01097252,  4.64565885])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.98078236,  0.01288857,  0.19467884],
         [ 0.01263319,  0.99152653, -0.12928864],
         [-0.19469558,  0.12926343,  0.9723089 ]]),
  array([-1.98766746, -0.88355057,  4.54431177])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 9.91496441e-01,  9.08900698e-03,  1.29816017e-01],
         [-6.01057883e-04,  9.97867165e-01, -6.52744943e-02],
         [-1.30132421e-01,  6.46414018e-02,  9.89387205e-01]]),
  array([-1.57792805, -0.65281097,  4.32611537])),
 (array([[581.78773191,   0.        , 490.25      ],
         [  0.        , 581.78773191, 272.75      ],
         [  0.        ,   0.        ,   1.        ]]),
  array([[ 0.99691859,  0.00879406,  0.07794859],
         [-0.00829108,  0.99994268, -0.00677394],
         [-0.0780037 ,  0.00610678,  0.99693437]]),
  array([-1.15893849, -0.45901755,  3.91884407]))]