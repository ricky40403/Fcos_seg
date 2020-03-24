from yacs.config import CfgNode as CN

__Fcos = CN()

##################################################
# Input
##################################################

__Fcos.INPUT = CN()

__Fcos.INPUT.MIN_SIZE_TRAIN = (800,)

__Fcos.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)

__Fcos.INPUT.MAX_SIZE_TRAIN = 1333

__Fcos.INPUT.MIN_SIZE_TEST = 800

__Fcos.INPUT.MAX_SIZE_TEST = 1333



# Values to be used for image normalization
__Fcos.INPUT.PIXEL_MEAN = [103.53, 116.28, 123.675]
# Values to be used for image normalization
__Fcos.INPUT.PIXEL_STD = [57.375, 57.12, 58.395]
# Convert image to BGR format (for Caffe2 models), in range 0-255
__Fcos.INPUT.TO_BGR255 = True


##################################################
# Model
##################################################

__Fcos.MODEL = CN()

# bg + 80 class
__Fcos.MODEL.NUM_CLASS = 81

__Fcos.MODEL.INF  = 1000000000

__Fcos.MODEL.DIV  = 32

__Fcos.MODEL.CENTERNESS_ON_REG = True

__Fcos.MODEL.NORM_REG = True

__Fcos.MODEL.HAED_RANGE = 4


# BackBone
__Fcos.MODEL.BACKBONE = "resnet50"


# FPN
__Fcos.MODEL.FPN = CN()

__Fcos.MODEL.FPN.EXTRA_NUM = 2

__Fcos.MODEL.FPN.OUT_C = 256


# Loss
__Fcos.MODEL.LOSS = CN()

__Fcos.MODEL.LOSS.FOCAL_GAMMA = 2.0

__Fcos.MODEL.LOSS.FOCAL_ALPHA  = 0.25

__Fcos.MODEL.LOSS.FOCAL_LOSS_BCE  = True

__Fcos.MODEL.LOSS.IOU_LOSS_TYPE  = "iou"


__Fcos.TRAIN = CN()

__Fcos.TRAIN.WORKER = 4

__Fcos.TRAIN.BATCH = 16

__Fcos.TRAIN.LR = 0.01

__Fcos.TRAIN.MOMENTUM = 0.9

__Fcos.TRAIN.EPOCH = 24

__Fcos.TRAIN.MILESTONES = [16, 22]


__Fcos.TEST = CN()

__Fcos.TEST.WORKER = 4

__Fcos.TEST.PRE_THRES = 0.05

__Fcos.TEST.NMS_THRES = 0.6

__Fcos.TEST.TOP_N = 1000

__Fcos.TEST.POST_TOP_N = 100






def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return __Fcos.clone()
