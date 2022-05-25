def add_brainSeg_config(cfg):
    cfg.MODEL.BACKBONE.NAME = 'BrainSegBackbone'
    
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[2,4,8], [4,8,16], [8,16,32], [16,32,64]]
    cfg.MODEL.RPN.IN_FEATURES = ['level1','level2', 'level3', 'level4']
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['level1','level2', 'level3', 'level4']
    cfg.MODEL.PIXEL_MEAN= [0,0,0,0]
    cfg.MODEL.PIXEL_STD = [1,1,1,1]
    cfg.MODEL.BRAINSEG.fl_inChannels = [1, 32, 64, 128]
    cfg.MODEL.BRAINSEG.fl_outChannels = [32, 64, 128, 256]
    cfg.MODEL.BRAINSEG.fl_lateral_channel = 64
    cfg.MODEL.BRAINSEG.norm = 'GN'