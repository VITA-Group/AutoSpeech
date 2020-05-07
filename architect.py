import torch


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, cfg):
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=cfg.TRAIN.ARCH_LR, betas=(0.5, 0.999), weight_decay=cfg.TRAIN.ARCH_WD)

  def step(self, input_valid, target_valid):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()