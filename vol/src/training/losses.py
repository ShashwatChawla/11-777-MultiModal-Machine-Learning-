import torch

def compute_rotation_loss(predicted, target):
    # Compute the rotation loss between predicted and target

    #TODO: Find a better loss with SO(3) constraint
    rotation_loss = torch.nn.functional.mse_loss(predicted, target)
    return rotation_loss

def compute_translation_loss(predicted, target):
    # Compute the translation loss between predicted and target
    translation_loss = torch.nn.functional.mse_loss(predicted, target)
    return translation_loss

def compute_loss(predicted_rotation, target_rotation, predicted_translation, target_translation, alpha):
    # Compute the weighted loss
    rotation_loss = compute_rotation_loss(predicted_rotation, target_rotation)
    translation_loss = compute_translation_loss(predicted_translation, target_translation)
    total_loss = alpha * rotation_loss + (alpha) * translation_loss

    return {"rotation_loss": rotation_loss, "translation_loss": translation_loss, "total_loss": total_loss}