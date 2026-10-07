from types import SimpleNamespace

from intentbench.finetune import loss_history


def test_loss_history_merges_train_and_eval_logs_per_epoch():
    log = [
        {"epoch": 1.0, "loss": 2.0},
        {"epoch": 1.0, "eval_loss": 1.5, "eval_macro_f1": 0.8},
        {"epoch": 2.0, "loss": 0.9},
        {"epoch": 2.0, "eval_loss": 0.7, "eval_macro_f1": 0.9},
        {"epoch": 2.0, "train_runtime": 10.0},  # end-of-training summary, has no loss
    ]
    trainer = SimpleNamespace(state=SimpleNamespace(log_history=log))
    assert loss_history(trainer) == [
        {"epoch": 1, "train_loss": 2.0, "loss": 1.5, "macro_f1": 0.8},
        {"epoch": 2, "train_loss": 0.9, "loss": 0.7, "macro_f1": 0.9},
    ]
