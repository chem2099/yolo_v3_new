from models.yolo import net
from models.yolo.YOLOv3 import YOLOv3

class Student:
    def __init__(self, inputs_x, net_name, batch_size):
        print('==> Init Teacher')
        self._model = net.__dict__[net_name](inputs_x, True)
        self._yolo_v3 = YOLOv3(self._model, None, batch_size=batch_size, is_training=True)
        print('==> Finished!')

    def get_scale_list(self):
        return self._yolo_v3.scales