from detectron2.engine.hooks import HookBase
from detectron2.evaluation.testing import flatten_results_dict
import detectron2.utils.comm as comm
import torch
import numpy as np

# Hook Function
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, logger=None):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.logger = logger

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        losses = []

        for idx, inputs in enumerate(self._data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            loss_batch = self._get_loss(inputs)
            if idx == 0:
                loss_dict = {}

                for key in list(loss_batch.keys()):
                    loss_dict[key] = loss_batch[key] / len(self._data_loader)

            else:
                for key in list(loss_batch.keys()):
                    loss_dict[key] += loss_batch[key] / len(self._data_loader)

        if self.logger is not None:
            loss_tot = 0.
            for key in list(loss_dict.keys()):
                name = 'loss/%s' %key
                self.logger[name].log(loss_dict[key])

                loss_tot += loss_dict[key]

            self.logger['loss/loss_tot'].log(loss_tot)

        self.trainer.storage.put_scalar('validation_loss', loss_tot)
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        return metrics_dict

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, logger):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self.logger = logger

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)

            for key in list(flattened_results.keys()):
                result = flattened_results[key]
                self.logger[key.replace('bbox', 'result')].log(result)
            
            self.logger['iter'].log(self.trainer.iter)

            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):

        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func

class IterHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still after the last iteration).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
    def after_step(self):
        self.trainer.model.module.iter = self.trainer.iter