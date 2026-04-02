try:
    from .model import LlavaQwenForCausalLM
except Exception:
    LlavaQwenForCausalLM = None

try:
    from .train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
except Exception:
    LazySupervisedDataset = None
    DataCollatorForSupervisedDataset = None
