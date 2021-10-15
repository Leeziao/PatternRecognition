from tensorboardX import SummaryWriter

class ScalerRecorder:
    def __init__(self, writer:SummaryWriter, name:str='Loss') -> None:
        self.writer = writer
        self.name = name
        self.total_step = 0
    def __call__(self, scaler):
        self.writer.add_scalar(self.name, scaler, self.total_step)
        self.total_step += 1