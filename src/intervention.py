import numpy as np


def generate_intervention(sequence):

    triggers = []

    lms = sequence[:, 0]
    delay = sequence[:, 1]
    attendance = sequence[:, 2]
    sentiment = sequence[:, 3]

    # Behavioural trend checks
    if lms[-1] < lms[0] * 0.7:
        triggers.append("Significant drop in LMS activity")

    if delay[-1] > delay.mean() * 1.5:
        triggers.append("Increasing assignment submission delays")

    if attendance[-1] < 75:
        triggers.append("Declining attendance rate")

    if sentiment.mean() < 0:
        triggers.append("Consistently negative sentiment")

    if len(triggers) == 0:
        triggers.append("No major behavioural risk indicators detected")

    return triggers


def recommend_action(triggers):

    if "Significant drop in LMS activity" in triggers:
        return "Schedule academic mentoring session"

    if "Declining attendance rate" in triggers:
        return "Notify faculty advisor and monitor attendance"

    if "Consistently negative sentiment" in triggers:
        return "Refer student to counseling support"

    if "Increasing assignment submission delays" in triggers:
        return "Send deadline reminder and study planning resources"

    return "No immediate intervention required"