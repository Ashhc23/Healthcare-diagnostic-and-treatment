#!/usr/bin/env python3
import random
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Constants for mapping
SYMPTOMS_TO_CONDITIONS: Dict[str, List[str]] = {
    'fever': ['Flu', 'COVID-19'],
    'cough': ['Bronchitis', 'COVID-19'],
    'fatigue': ['Anemia', 'Flu'],
}

TREATMENTS: Dict[str, List[str]] = {
    'Flu': ['Rest', 'Hydration', 'Paracetamol'],
    'COVID-19': ['Isolation', 'Antiviral medication', 'Oxygen therapy'],
    'Bronchitis': ['Cough syrup', 'Antibiotics', 'Steam inhalation'],
    'Anemia': ['Iron supplements', 'Dietary adjustments'],
}


def diagnose(symptoms: List[str]) -> List[Tuple[str, int]]:
    """
    Count possible conditions based on reported symptoms.

    Args:
        symptoms: List of symptom keywords.

    Returns:
        Sorted list of tuples (condition, score) descending by likelihood.
    """
    scores: Dict[str, int] = {}
    for symptom in symptoms:
        for condition in SYMPTOMS_TO_CONDITIONS.get(symptom, []):
            scores[condition] = scores.get(condition, 0) + 1

    # Sort by score descending, then alphabetically
    sorted_conditions = sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )
    return sorted_conditions


def suggest_treatment(condition: str) -> List[str]:
    """
    Return recommended treatment steps for the top condition.
    """
    return TREATMENTS.get(condition, ['Consult a healthcare professional'])


def generate_iot_data(minutes: int = 10) -> pd.DataFrame:
    """
    Simulate IoT health vitals over a time window.

    Args:
        minutes: Number of minutes to simulate.

    Returns:
        DataFrame with Time, Temperature, Heart Rate, and Oxygen Level.
    """
    base_time = datetime.now()
    times = [base_time + timedelta(minutes=i) for i in range(minutes)]
    data = {
        'Time': times,
        'Temperature (°C)': [round(random.uniform(36.5, 38.5), 2) for _ in range(minutes)],
        'Heart Rate (bpm)': [random.randint(60, 100) for _ in range(minutes)],
        'Oxygen Level (%)': [random.randint(92, 100) for _ in range(minutes)]
    }
    return pd.DataFrame(data)


def plot_vitals(df: pd.DataFrame) -> None:
    """
    Plot simulated IoT health metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Time'], df['Temperature (°C)'], label='Temperature (°C)', marker='o')
    ax.plot(df['Time'], df['Heart Rate (bpm)'], label='Heart Rate (bpm)', marker='x')
    ax.plot(df['Time'], df['Oxygen Level (%)'], label='Oxygen Level (%)', marker='s')

    ax.set_title('Real-time IoT Health Metrics')
    ax.set_xlabel('Time')
    ax.set_ylabel('Measurements')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Simulate diagnosis and IoT health monitoring.'
    )
    parser.add_argument(
        '-s', '--symptoms', nargs='+', default=['fever', 'cough', 'fatigue'],
        help='List of patient symptoms.'
    )
    parser.add_argument(
        '-m', '--minutes', type=int, default=12,
        help='Duration in minutes for IoT simulation.'
    )
    args = parser.parse_args()

    logging.info(f"Received symptoms: {args.symptoms}")

    results = diagnose(args.symptoms)
    if results:
        logging.info("Diagnosis results (most likely first):")
        for condition, score in results:
            logging.info(f"- {condition}: score {score}")

        top_condition = results[0][0]
        treatment_steps = suggest_treatment(top_condition)
        logging.info(f"Suggested treatment for {top_condition}: {', '.join(treatment_steps)}")
    else:
        logging.warning("No conditions matched the provided symptoms.")

    df = generate_iot_data(args.minutes)
    logging.info("Sample IoT data:\n%s", df.head())
    plot_vitals(df)


if __name__ == '__main__':
    main()
