"""Voting ensemble used in the paper (majority vote for gender, mean for age)."""
import numpy as np


def ensemble(gender_probs, ages):
    """
    gender_probs, ages: arrays of shape (n_models, n_samples).
    gender_probs holds P(female) from each model.
    Returns (is_female[bool], confidence[0-1], age[float]) each of shape (n_samples,).
    """
    gender_probs = np.asarray(gender_probs, dtype="float32")
    ages = np.asarray(ages, dtype="float32")
    n_models = gender_probs.shape[0]

    votes = (gender_probs >= 0.5).sum(axis=0)
    is_female = np.where(
        votes * 2 == n_models,                  # tie (possible with 2 models)
        gender_probs.mean(axis=0) >= 0.5,       # fall back to mean probability
        votes * 2 > n_models,                   # normal majority vote
    )
    confidence = np.where(is_female, gender_probs, 1.0 - gender_probs).mean(axis=0)
    age = ages.mean(axis=0)
    return is_female, confidence, age
