from detectron2.evaluation import DatasetEvaluator
from collections import defaultdict
import numpy as np

class DiceEval(DatasetEvaluator):
    def __init__(self, metadata) -> None:
        super().__init__()
        self.classes_ids = metadata.thing_dataset_id_to_contiguous_id.values()
        self.classes_names = metadata.thing_classes


    def reset(self):
        self.dice_score = defaultdict(list)
        self.count = 0

    def process(self, inputs, outputs):
        
        for gt , pred in zip(inputs, outputs):
          gt['instances'] = gt['instances'].to('cpu')
          pred['instances'] = pred['instances'].to('cpu')
          # print(f'inputs-------------- {gt}')
          
          if (len(gt['instances']) == 0) and (len(pred['instances']) != 0):
            # if there is no instances in the input image and the model predicted some classes, we set the dice score of predicted classes to 0
            
            for id in np.unique(pred['instances'].pred_classes):
              class_name = self.classes_names[id]
              self.dice_score[class_name].append( 0 )
            continue

          elif  (len(gt['instances']) == 0) and (len(pred['instances']) == 0):
            # we reward the model for not predicting no false positives
            
            for id in self.classes_ids:
              class_name = self.classes_names[id]
              self.dice_score[class_name].append( 1 )
            continue

          
          gt_classes = gt['instances'].gt_classes
          # not sure what is the order of box coordinates so we repeate the image width because curentely working with square images
          boxes = torch.tensor([
                                [0, 0, gt['height'], gt['height']] for i in range(len(gt_classes))
                                ])
          gt_masks = gt['instances'].gt_masks.crop_and_resize(boxes, gt['height'])
          
          pred_classes = pred['instances'].pred_classes
          pred_masks = pred['instances'].pred_masks
          
          for class_id in self.classes_ids:

            class_name = self.classes_names[class_id]

            gt_indices = np.where(gt_classes == class_id)[0]

            pred_indices = np.where(pred_classes == class_id)[0]
            # print(f'--------------{gt_indices.any()}')
            if (gt_indices.size == 0) and (pred_indices.size == 0):
              # if a class isn't in gt and pred then we reward the model fot not giving a false positive
              
              self.dice_score[class_name].append( 1 )
              continue

            if ((gt_indices.size != 0) and  (pred_indices.size == 0)) or ((gt_indices.size == 0) and  (pred_indices.size != 0)):
              # if a class is in gt and not in pred we set the score to 0 and vise versa
              
              self.dice_score[class_name].append( 0 )
              continue
            
            gt_mask = np.add.reduce(gt_masks[gt_indices, :, :].numpy(),axis=0).astype(bool)
            pred_mask = np.add.reduce(pred_masks[pred_indices, :, :].numpy(),axis=0).astype(bool)
            
            self.dice_score[class_name].append( self.single_dice_coef(gt_mask, pred_mask) )
            self.count += 1




    def evaluate(self):
        # save self.count somewhere, or print it, or return it.
        print('Dice score results')
        return {key : np.mean(value) for key, value in self.dice_score.items()}

    def single_dice_coef(self, y_true, y_pred_bin):
        # shape of y_true and y_pred_bin: (height, width)
        intersection = np.sum(y_true * y_pred_bin)
        if (np.sum(y_true)==0) and (np.sum(y_pred_bin)==0):
            return 1
        return (2*intersection) / (np.sum(y_true) + np.sum(y_pred_bin))