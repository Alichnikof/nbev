from peft import LoraConfig, get_peft_model


def setup_finetuning(backbone, lora_config, lora_target_regex, full_finetune=False):
    """Decides which backbone weights train. The head always trains.

    full_finetune=True        -> every backbone weight trains (no LoRA)
    lora.last_k_layers == 0   -> frozen backbone, only the head trains
    lora.last_k_layers == k   -> LoRA adapters on the attention projections of the last k layers
    """
    if full_finetune:
        return backbone
    if lora_config["last_k_layers"] == 0:
        backbone.requires_grad_(False)
        return backbone
    return get_peft_model(backbone, LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_target_regex,
    ))
