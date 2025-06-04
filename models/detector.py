import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detector_model(num_classes):
    # Загрузка базовой модели с предобученными весами
    model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')

    # Получаем размерность входов последнего слоя
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Заменяем предсказатель на наш, с нужным числом классов (включая фон)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
