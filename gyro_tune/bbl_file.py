import struct
import os
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.stats import zscore
from tabulate import tabulate
import re


def parse_cli_dump(file_path):
    """Parse a Betaflight CLI dump file and extract advanced gyro and rate settings."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    current_values = {
        "gyro_hardware_lpf": None,
        "gyro_lpf1_type": None,
        "gyro_lpf1_static_hz": None,
        "gyro_lpf2_type": None,
        "gyro_lpf2_static_hz": None,
        "gyro_notch1_hz": None,
        "gyro_notch1_cutoff": None,
        "gyro_notch2_hz": None,
        "gyro_notch2_cutoff": None,
        "gyro_calib_duration": None,
        "gyro_calib_noise_limit": None,
        "gyro_offset_yaw": None,
        "gyro_overflow_detect": None,
        "yaw_spin_recovery": None,
        "yaw_spin_threshold": None,
        "gyro_to_use": None,
        "dyn_notch_count": None,
        "dyn_notch_q": None,
        "dyn_notch_min_hz": None,
        "dyn_notch_max_hz": None,
        "gyro_lpf1_dyn_min_hz": None,
        "gyro_lpf1_dyn_max_hz": None,
        "gyro_lpf1_dyn_expo": None,
        "gyro_filter_debug_axis": None,
        "roll_rc_rate": None,
        "pitch_rc_rate": None,
        "yaw_rc_rate": None,
        "roll_expo": None,
        "pitch_expo": None,
        "yaw_expo": None,
        "roll_srate": None,
        "pitch_srate": None,
        "yaw_srate": None,
        "throttle_limit_type": None,
        "throttle_limit_percent": None,
        "roll_rate_limit": None,
        "pitch_rate_limit": None,
        "yaw_rate_limit": None,
    }

    with open(file_path, "r") as file:
        dump_lines = file.readlines()

        # Look for the relevant "set" commands and extract values
        for line in dump_lines:
            # Extract gyro filter settings
            if "gyro_hardware_lpf" in line:
                match = re.search(r"set gyro_hardware_lpf = (\w+)", line)
                if match:
                    current_values["gyro_hardware_lpf"] = match.group(1)

            elif "gyro_lpf1_type" in line:
                match = re.search(r"set gyro_lpf1_type = (\w+)", line)
                if match:
                    current_values["gyro_lpf1_type"] = match.group(1)

            elif "gyro_lpf1_static_hz" in line:
                match = re.search(r"set gyro_lpf1_static_hz = (\d+)", line)
                if match:
                    current_values["gyro_lpf1_static_hz"] = int(match.group(1))

            elif "gyro_lpf2_type" in line:
                match = re.search(r"set gyro_lpf2_type = (\w+)", line)
                if match:
                    current_values["gyro_lpf2_type"] = match.group(1)

            elif "gyro_lpf2_static_hz" in line:
                match = re.search(r"set gyro_lpf2_static_hz = (\d+)", line)
                if match:
                    current_values["gyro_lpf2_static_hz"] = int(match.group(1))

            elif "gyro_notch1_hz" in line:
                match = re.search(r"set gyro_notch1_hz = (\d+)", line)
                if match:
                    current_values["gyro_notch1_hz"] = int(match.group(1))

            elif "gyro_notch1_cutoff" in line:
                match = re.search(r"set gyro_notch1_cutoff = (\d+)", line)
                if match:
                    current_values["gyro_notch1_cutoff"] = int(match.group(1))

            elif "gyro_notch2_hz" in line:
                match = re.search(r"set gyro_notch2_hz = (\d+)", line)
                if match:
                    current_values["gyro_notch2_hz"] = int(match.group(1))

            elif "gyro_notch2_cutoff" in line:
                match = re.search(r"set gyro_notch2_cutoff = (\d+)", line)
                if match:
                    current_values["gyro_notch2_cutoff"] = int(match.group(1))

            # Extract other gyro and PID related settings
            elif "gyro_calib_duration" in line:
                match = re.search(r"set gyro_calib_duration = (\d+)", line)
                if match:
                    current_values["gyro_calib_duration"] = int(match.group(1))

            elif "gyro_calib_noise_limit" in line:
                match = re.search(r"set gyro_calib_noise_limit = (\d+)", line)
                if match:
                    current_values["gyro_calib_noise_limit"] = int(match.group(1))

            elif "gyro_offset_yaw" in line:
                match = re.search(r"set gyro_offset_yaw = (\d+)", line)
                if match:
                    current_values["gyro_offset_yaw"] = int(match.group(1))

            elif "gyro_overflow_detect" in line:
                match = re.search(r"set gyro_overflow_detect = (\w+)", line)
                if match:
                    current_values["gyro_overflow_detect"] = match.group(1)

            elif "yaw_spin_recovery" in line:
                match = re.search(r"set yaw_spin_recovery = (\w+)", line)
                if match:
                    current_values["yaw_spin_recovery"] = match.group(1)

            elif "yaw_spin_threshold" in line:
                match = re.search(r"set yaw_spin_threshold = (\d+)", line)
                if match:
                    current_values["yaw_spin_threshold"] = int(match.group(1))

            elif "gyro_to_use" in line:
                match = re.search(r"set gyro_to_use = (\w+)", line)
                if match:
                    current_values["gyro_to_use"] = match.group(1)

            # Extract rate and limit settings
            elif "roll_rc_rate" in line:
                match = re.search(r"set roll_rc_rate = (\d+)", line)
                if match:
                    current_values["roll_rc_rate"] = int(match.group(1))

            elif "pitch_rc_rate" in line:
                match = re.search(r"set pitch_rc_rate = (\d+)", line)
                if match:
                    current_values["pitch_rc_rate"] = int(match.group(1))

            elif "yaw_rc_rate" in line:
                match = re.search(r"set yaw_rc_rate = (\d+)", line)
                if match:
                    current_values["yaw_rc_rate"] = int(match.group(1))

            elif "roll_expo" in line:
                match = re.search(r"set roll_expo = (\d+)", line)
                if match:
                    current_values["roll_expo"] = int(match.group(1))

            elif "pitch_expo" in line:
                match = re.search(r"set pitch_expo = (\d+)", line)
                if match:
                    current_values["pitch_expo"] = int(match.group(1))

            elif "yaw_expo" in line:
                match = re.search(r"set yaw_expo = (\d+)", line)
                if match:
                    current_values["yaw_expo"] = int(match.group(1))

            elif "roll_srate" in line:
                match = re.search(r"set roll_srate = (\d+)", line)
                if match:
                    current_values["roll_srate"] = int(match.group(1))

            elif "pitch_srate" in line:
                match = re.search(r"set pitch_srate = (\d+)", line)
                if match:
                    current_values["pitch_srate"] = int(match.group(1))

            elif "yaw_srate" in line:
                match = re.search(r"set yaw_srate = (\d+)", line)
                if match:
                    current_values["yaw_srate"] = int(match.group(1))

            # Extract throttle limit and rate limits
            elif "throttle_limit_type" in line:
                match = re.search(r"set throttle_limit_type = (\w+)", line)
                if match:
                    current_values["throttle_limit_type"] = match.group(1)

            elif "throttle_limit_percent" in line:
                match = re.search(r"set throttle_limit_percent = (\d+)", line)
                if match:
                    current_values["throttle_limit_percent"] = int(match.group(1))

            elif "roll_rate_limit" in line:
                match = re.search(r"set roll_rate_limit = (\d+)", line)
                if match:
                    current_values["roll_rate_limit"] = int(match.group(1))

            elif "pitch_rate_limit" in line:
                match = re.search(r"set pitch_rate_limit = (\d+)", line)
                if match:
                    current_values["pitch_rate_limit"] = int(match.group(1))

            elif "yaw_rate_limit" in line:
                match = re.search(r"set yaw_rate_limit = (\d+)", line)
                if match:
                    current_values["yaw_rate_limit"] = int(match.group(1))

    return current_values


file_path = "betaflight_dump.txt"
try:
    current_values = parse_cli_dump(file_path)
except Exception as e:
    print("Error:", e)


def high_pass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


GYRO_X_ID = 1
GYRO_Y_ID = 2
GYRO_Z_ID = 3


def plot_noise(bbl_df):

    plt.figure(figsize=(10, 5))
    plt.plot(bbl_df.index, bbl_df["Value"], label="Original Data", alpha=0.6)
    plt.plot(bbl_df.index, bbl_df["Filtered_Value"], label="Filtered Data", alpha=0.8)
    plt.scatter(
        bbl_df.index[bbl_df["Noise_Flag"]],
        bbl_df["Value"][bbl_df["Noise_Flag"]],
        color="red",
        label="Noise",
        marker="x",
    )
    plt.legend()
    plt.title("Gyro Data with Noise Filtering")
    plt.show()


def plot_noise_scatter(bbl_df):
    # Plot Data
    plt.figure(figsize=(10, 5))
    plt.plot(bbl_df.index, bbl_df["Value"], label="Original Data", alpha=0.6)
    plt.scatter(
        bbl_df.index[bbl_df["Noise_Flag"]],
        bbl_df.loc[bbl_df["Noise_Flag"], "Value"],
        color="red",
        label="Noise",
        marker="x",
    )
    plt.legend()
    plt.title("Gyro Data with Noise Detection")
    plt.show()


# Run Analysis
def parse_bbl_file(file_path):
    """
    Parses a .BBL file and extracts its contents into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        header = file.read(16)

        version, entry_count = struct.unpack("<II", header[:8])

        print(f"BBL Version: {version}")
        print(f"Number of Entries: {entry_count}")

        entries = []
        for _ in range(entry_count):
            # Assuming each entry has an ID (int) and a value (float)
            entry_data = file.read(8)  # Adjust based on format
            if len(entry_data) < 8:
                break
            entry_id, value = struct.unpack("<If", entry_data)
            entries.append((entry_id, value))

        df = pd.DataFrame(entries, columns=["Entry ID", "Value"])
        return df


def suggest_flight_style_settings(current_values, noise_level):
    """Suggest Betaflight CLI settings for cinematic and freestyle, adjusted for noise level."""
    base_settings = {
        "gyro_hardware_lpf": "NORMAL",
        "gyro_lpf1_type": "PT1",
        "gyro_lpf2_type": "PT1",
        "dyn_notch_count": 3,
        "dyn_notch_q": 500 if noise_level < 1.5 else 400,
        "dyn_notch_min_hz": 150,
        "dyn_notch_max_hz": 600,
        "throttle_limit_type": "SCALE",
    }

    # Cinematic settings (smooth, stable footage)
    cinematic_settings = {
        "gyro_lpf1_static_hz": (
            80 if noise_level > 1.5 else 90
        ),  # Less filtering with low noise
        "gyro_lpf2_static_hz": 60 if noise_level > 1.5 else 70,
        "gyro_notch1_hz": 200,
        "gyro_notch1_cutoff": 180,
        "gyro_notch2_hz": 0,  # Disable second notch with low noise
        "gyro_notch2_cutoff": 0,
        "gyro_calib_duration": 125,
        "gyro_calib_noise_limit": 3,
        "yaw_spin_recovery": "ON",
        "yaw_spin_threshold": 1950,
        "roll_rc_rate": 100,
        "pitch_rc_rate": 100,
        "yaw_rc_rate": 90,
        "roll_expo": 30,
        "pitch_expo": 30,
        "yaw_expo": 25,
        "roll_srate": 70,
        "pitch_srate": 70,
        "yaw_srate": 65,
        "roll_rate_limit": 199,
        "pitch_rate_limit": 199,
        "yaw_rate_limit": 190,
    }

    # Freestyle settings (responsive, agile control)
    freestyle_settings = {
        "gyro_lpf1_static_hz": (
            120 if noise_level > 1.5 else 130
        ),  # Minimal filtering with low noise
        "gyro_lpf2_static_hz": 100 if noise_level > 1.5 else 110,
        "gyro_notch1_hz": 200,
        "gyro_notch1_cutoff": 190,
        "gyro_notch2_hz": 150 if noise_level > 1.0 else 0,  # Optional second notch
        "gyro_notch2_cutoff": 140 if noise_level > 1.0 else 0,
        "gyro_calib_duration": 100,
        "gyro_calib_noise_limit": 5,
        "yaw_spin_recovery": "ON",
        "yaw_spin_threshold": 1950,
        "roll_rc_rate": 140,
        "pitch_rc_rate": 140,
        "yaw_rc_rate": 120,
        "roll_expo": 20,
        "pitch_expo": 20,
        "yaw_expo": 15,
        "roll_srate": 85,
        "pitch_srate": 85,
        "yaw_srate": 80,
        "roll_rate_limit": 250,
        "pitch_rate_limit": 250,
        "yaw_rate_limit": 220,
    }

    # Merge settings
    cinematic_suggestions = {**current_values, **base_settings, **cinematic_settings}
    freestyle_suggestions = {**current_values, **base_settings, **freestyle_settings}

    return cinematic_suggestions, freestyle_suggestions


def print_cli_settings(settings, style_name):
    """Print CLI commands for a given flight style."""
    print(f"\n### Suggested {style_name} Settings ###")
    print(
        "# Optimized for smooth footage"
        if style_name == "Cinematic"
        else "# Optimized for agile control"
    )
    for key, value in settings.items():
        if value is not None:
            print(f"set {key} = {value}")


# [Previous high_pass_filter function remains unchanged]


def analyze_gyro_data(df):
    """Extended analyze_gyro_data with flight style suggestions."""
    df.dropna(inplace=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df["Z-score"] = zscore(df["Value"])
    df["Noise_Flag"] = df["Z-score"].abs() > 3
    df["Filtered_Value"] = (
        high_pass_filter(df["Value"], cutoff=10, fs=1000, order=3)
        if len(df) > 5
        else df["Value"]
    )

    noise_level = df["Z-score"].abs().mean()

    # Get current CLI settings
    cli_file_path = "betaflight_dump.txt"
    try:
        current_values = parse_cli_dump(cli_file_path)
    except:
        current_values = {}  # Use empty dict if CLI dump unavailable

    # Get flight style suggestions
    cinematic_settings, freestyle_settings = suggest_flight_style_settings(
        current_values, noise_level
    )

    # Print current settings
    print("\n### Current Betaflight Settings ###")
    for setting, value in current_values.items():
        print(f"{setting}: {value}")

    # Print suggested settings
    print_cli_settings(cinematic_settings, "Cinematic")
    print_cli_settings(freestyle_settings, "Freestyle")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(
        df.index,
        df["Value"],
        label="Original Data",
        alpha=0.6,
        linestyle="--",
        color="gray",
    )
    plt.plot(df.index, df["Filtered_Value"], label="Filtered Data", color="blue")
    plt.scatter(
        df.index[df["Noise_Flag"]],
        df.loc[df["Noise_Flag"], "Value"],
        color="red",
        label="Noise",
        marker="x",
        s=50,
    )

    plt.xlabel("Time (Index)")
    plt.ylabel("Gyro Value")
    plt.title(f"Gyroscope Data Analysis (Noise Level: {noise_level:.2f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def generate_comparison_table(current_values, noise_level=2.0):
    """
    Generate a side-by-side comparison table of current and suggested settings.

    Args:
        cli_file_path (str): Path to the Betaflight CLI dump file
        noise_level (float): Noise level to influence suggestions (default 2.0)
    """
    # Get current and suggested settings
    cinematic_settings, freestyle_settings = suggest_flight_style_settings(
        current_values, noise_level
    )

    # Define the settings to compare (subset of all possible settings)
    settings_to_compare = [
        "gyro_hardware_lpf",
        "gyro_lpf1_type",
        "gyro_lpf1_static_hz",
        "gyro_lpf2_type",
        "gyro_lpf2_static_hz",
        "gyro_notch1_hz",
        "gyro_notch1_cutoff",
        "gyro_notch2_hz",
        "gyro_notch2_cutoff",
        "gyro_calib_duration",
        "gyro_calib_noise_limit",
        "yaw_spin_recovery",
        "yaw_spin_threshold",
        "dyn_notch_count",
        "dyn_notch_q",
        "dyn_notch_min_hz",
        "dyn_notch_max_hz",
        "roll_rc_rate",
        "pitch_rc_rate",
        "yaw_rc_rate",
        "roll_expo",
        "pitch_expo",
        "yaw_expo",
        "roll_srate",
        "pitch_srate",
        "yaw_srate",
        "throttle_limit_type",
        "roll_rate_limit",
        "pitch_rate_limit",
        "yaw_rate_limit",
    ]

    # Prepare table data
    table_data = []
    for setting in settings_to_compare:
        row = [
            setting,
            current_values.get(setting, "N/A"),
            cinematic_settings.get(setting, "N/A"),
            freestyle_settings.get(setting, "N/A"),
        ]
        table_data.append(row)

    # Headers for the table
    headers = ["Setting", "Current", "Cinematic", "Freestyle"]

    # Generate table using tabulate (if available)
    try:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback to basic string formatting
        print("\nSide-by-Side Comparison Table")
        print("-" * 80)
        print(f"{'Setting':<25} {'Current':<15} {'Cinematic':<15} {'Freestyle':<15}")
        print("-" * 80)
        for row in table_data:
            print(f"{row[0]:<25} {str(row[1]):<15} {str(row[2]):<15} {str(row[3]):<15}")
        print("-" * 80)


def main():
    file_path = "flight_log.bbl"
    try:
        bbl_df = parse_bbl_file(file_path)
        # print(bbl_df.isna().sum())  # Check for missing values
        bbl_df = bbl_df.dropna()  # Remove rows with NaN
        bbl_df["Value"].fillna(0, inplace=True)
        bbl_df["Value"] = pd.to_numeric(bbl_df["Value"], errors="coerce")
        noise_level = bbl_df["Value"].std()
        # print(f"Estimated Noise Level (STD): {noise_level}")
        bbl_df["Rolling_STD"] = bbl_df["Value"].rolling(window=10).std()
        bbl_df["Rolling_Mean"] = bbl_df["Value"].rolling(window=10).mean()
        bbl_df["Filtered_Value"] = high_pass_filter(bbl_df["Value"])
        bbl_df["Z-score"] = zscore(bbl_df["Value"])
        bbl_df["Noise_Flag"] = bbl_df["Z-score"].abs() > 3

        # print(bbl_df.head(100))
        file_path = "betaflight_dump.txt"
        try:
            current_values = parse_cli_dump(file_path)
            generate_comparison_table(current_values, noise_level)

        except Exception as e:
            print("Error:", e)
        # plot_noise(bbl_df)
        analyze_gyro_data(bbl_df)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
