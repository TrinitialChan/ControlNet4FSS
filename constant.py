
def coco_mapper():
    coco_to_ade2k = {
    0:0  ,#__background__
    1:12  ,#person
    2:127  ,#bicycle
    3:20  ,#car
    4:116  ,#motorcycle
    5:90  ,#airplane
    6:80  ,#bus
    7:20  ,#train
    8:83  ,#truck
    9:76  ,#boat
    10:136  ,#traffic light
    11:6  ,#fire hydrant ?
    12:43  ,#stop sign
    13:6  ,#parking meter ?
    14:69  ,#bench
    15:126  ,#bird
    16:126  ,#cat
    17:126  ,#dog
    18:126  ,#horse
    19:126  ,#sheep
    20:126  ,#cow
    21:126  ,#elephant
    22:126  ,#bear
    23:126  ,#zebra
    24:126  ,#giraffe
    25:115  ,#backpack
    26:108  ,#umbrella ?
    27:115  ,#handbag
    28:92  ,#tie
    29:41  ,#suitcase
    30:108  ,#frisbee
    31:108  ,#skis
    32:108  ,#snowboard
    33:108  ,#sports ball
    34:108  ,#kite
    35:108  ,#baseball bat
    36:108  ,#baseball glove
    37:108  ,#skateboard
    38:108  ,#surfboard
    39:108  ,#tennis racket
    40:98  ,#bottle
    41:147  ,#wine glass
    42:108  ,#cup
    43:108  ,#fork
    44:108  ,#knife
    45:108  ,#spoon
    46:108  ,#bowl
    47:120  ,#banana
    48:120  ,#apple
    49:120  ,#sandwich
    50:120  ,#orange
    51:120  ,#broccoli
    52:120  ,#carrot
    53:120  ,#hot dog
    54:120  ,#pizza
    55:120  ,#donut
    56:120  ,#cake
    57:19  ,#chair
    58:23  ,#couch
    59:17  ,#potted plant
    60:7  ,#bed
    61:15  ,#dining table
    62:65  ,#toilet
    63:89  ,#tv
    64:74  ,#laptop
    65:74  ,#mouse
    66:74  ,#remote
    67:74  ,#keyboard
    68:74  ,#cell phone
    69:124  ,#microwave
    70:118  ,#oven
    71:71  ,#toaster
    72:47  ,#sink
    73:50  ,#refrigerator
    74:67  ,#book
    75:148 ,#clock
    76:135  ,#vase
    77:108  ,#scissors
    78:108  ,#teddy bear
    79:108  ,#hair drier ?
    80:108  #toothbrush ?
}
    return coco_to_ade2k

def coco_name():
    coco_table = {
    0:'background',
    1:'person',
    2:'bicycle',
    3:'car',
    4:'motorcycle',
    5:'airplane',
    6:'bus',
    7:'train',
    8:'truck',
    9:'boat',
    10:'traffic light',
    11:'fire hydrant',
    12:'stop sign',
    13:'parking meter',
    14:'bench',
    15:'bird',
    16:'cat',
    17:'dog',
    18:'horse',
    19:'sheep',
    20:'cow',
    21:'elephant',
    22:'bear',
    23:'zebra',
    24:'giraffe',
    25:'backpack',
    26:'umbrella',
    27:'handbag',
    28:'tie',
    29:'suitcase',
    30:'frisbee',
    31:'skis',
    32:'snowboard',
    33:'sports ball',
    34:'kite',
    35:'baseball bat',
    36:'baseball glove',
    37:'skateboard',
    38:'surfboard',
    39:'tennis racket',
    40:'bottle',
    41:'wine glass',
    42:'cup',
    43:'fork',
    44:'knife',
    45:'spoon',
    46:'bowl',
    47:'banana',
    48:'apple',
    49:'sandwich',
    50:'orange',
    51:'broccoli',
    52:'carrot',
    53:'hot dog',
    54:'pizza',
    55:'donut',
    56:'cake',
    57:'chair',
    58:'couch',
    59:'potted plant',
    60:'bed',
    61:'dining table',
    62:'toilet',
    63:'tv',
    64:'laptop',
    65:'computer mouse',
    66:'remote controller',
    67:'keyboard',
    68:'cell phone',
    69:'microwave',
    70:'oven',
    71:'toaster',
    72:'sink',
    73:'refrigerator',
    74:'book',
    75:'clock',
    76:'vase',
    77:'scissors',
    78:'teddy bear',
    79:'hair drier',
    80:'toothbrush'
}
    return coco_table
def pascal_name():
    pascal_table = ["background",
            "airplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "dining table", "dog", "horse", "motorbike", "person",
            "potted plant", "sheep", "sofa", "train", "television"
            ]
    return pascal_table

def pascal_mapper():
    pascal_to_aed2k = {
        0:0,
        1:90,
        2:127,
        3:126, # animal
        4:76,
        5:98,
        6:80,
        7:20,
        8:126, #animal
        9:19,
        10:126, #animal
        11:15,
        12:126, #animal
        13:126, #animal
        14:116,
        15:12,
        16:17, #plant is 17 ,but flower pot is 126
        17:126, #animal
        18:23,
        19:20, #car;auto;automobile;machine;motorcar
        20:89,
    }
    return pascal_to_aed2k

def fss_name():
    with open('./list/fss1k_all.txt', 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    fss_list = [data.split('#')[0] for data in fold_n_metadata]
    return fss_list

def fss_mapper():
    with open('./list/fss2ade.txt', 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
    fss2ade = [int(data.split('#')[-1]) for data in fold_n_metadata]
    return fss2ade