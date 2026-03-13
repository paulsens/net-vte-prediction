from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

class Scheduler():
    def __init__(self, config):
        self.config=config

    def cosine_flat(self, optimizer):
        scheduler1 = CosineAnnealingLR(optimizer, T_max=self.config['milestones'][0], eta_min=.1*self.config['lr'])
        scheduler2 = LinearLR(optimizer, start_factor=.1, end_factor=.1)
        return SequentialLR(optimizer, [scheduler1, scheduler2], milestones=self.config['milestones'])

    def cosine_simple(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.config['T_max'], eta_min=.1*self.config['lr'])

    def flat(self, optimizer):
        return LinearLR(optimizer, start_factor=1, end_factor=1)
