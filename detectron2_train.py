# scripts/detectron2_train.py
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

def train_detectron2(config_file, output_dir, num_classes=3, num_epochs=50):
    """
    Train Detectron2 model.

    Args:
        config_file (str): Path to the config file.
        output_dir (str): Directory to save model checkpoints.
        num_classes (int): Number of classes.
        num_epochs (int): Number of training epochs.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("banana_train",)
    cfg.DATASETS.TEST = ("banana_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Starting from pre-trained model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = num_epochs * 100  # Approximate, adjust based on dataset size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"Detectron2 model trained and saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    output_directory = "models/detectron2_output"
    train_detectron2(
        config_file=config_file,
        output_dir=output_directory,
        num_classes=3,
        num_epochs=50
    )
