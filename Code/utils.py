import matplotlib.pyplot as plt


def display_waveform(waveform, title, sr=8000, save_path=None):
    """
    Display waveform plot and optionally save it as an image.

    Args:
        waveform (numpy.ndarray or list): The waveform data to plot.
        title (str): The title of the plot.
        sr (int): The sampling rate of the audio. Default is 8000.
        save_path (str, optional): The path to save the image. If None, the image is not saved.
    """
    plt.figure()
    plt.plot(waveform, 'k', linewidth=1.0)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    # Save the image if a save path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300)  # Save as PNG with high resolution
        print(f"Waveform plot saved to {save_path}")

    # Display the plot
    plt.show()
