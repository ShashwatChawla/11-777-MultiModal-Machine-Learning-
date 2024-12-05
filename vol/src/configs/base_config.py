class BaseConfig:
    def __init__(
        self,
        model: torch.nn.Module,
        data_modalities: list[str]
    ):
        self.model = model
        self.data_modalities = data_modalities