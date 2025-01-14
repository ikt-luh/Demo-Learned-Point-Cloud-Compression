import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

matplotlib.use("template")
plt.rcParams.update({
    'font.family': 'serif',            # Use a serif font
    'savefig.dpi': 300,                # Resolution for saved figures
    'savefig.format': 'pdf',           # File format for saving figures
    'pdf.fonttype': 42,                # Use Type 42 (TrueType) fonts in PDF
    'ps.fonttype': 42                  # Use Type 42 (TrueType) fonts in PS
})

matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42

colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#ff9999", 
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb"
]

# Templates
logs_base_path = "./logs"
figures_path = "./figures"
highest_fps = 10
sequences = ["jacket"]


def plot():
    data = load_data()

    # Encoder and Decoder Latency
    plot_coding_times(data)
    plot_coding_times_vs_num_points(data)
    plot_end_to_end_latency(data)
# Load the data

def load_data(save=False):
    merged_results = []

    for fps in range(1, highest_fps + 1):
        for seq in sequences:
            sender_path = os.path.join(logs_base_path, "sender", f"{seq}_{fps}fps.csv")
            receiver_path = os.path.join(logs_base_path, "receiver", f"{seq}_{fps}fps.csv")

            sender_data = pd.read_csv(sender_path) if os.path.exists(sender_path) else pd.DataFrame()
            receiver_data = pd.read_csv(receiver_path) if os.path.exists(receiver_path) else pd.DataFrame()

            sender_data["fps"] = fps
            sender_data["sequence"] = seq
            receiver_data["fps"] = fps
            receiver_data["sequence"] = seq

            # Merge sender and receiver data on ID
            merged = pd.merge(
                sender_data,
                receiver_data,
                on="ID",
                how="outer",
                suffixes=("_sender", "_receiver")
            )
            merged["packet_received"] = ~merged.isnull().any(axis=1)

            merged_results.append(merged)

    final_data = pd.concat(merged_results, ignore_index=True)

    if save:
        final_data.to_csv("merged_data.csv", index=False)

    return final_data

def plot_coding_times(all_data):
    # Initialize dictionaries to store aggregated data
    encoder_timings = {seq: [] for seq in sequences}
    decoder_timings = {seq: [] for seq in sequences}
    
    for seq in sequences:
        for fps in range(1, highest_fps + 1):
            # Filter data for the specific sequence and FPS
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            
            # Preprocessing
            data["time_measurements_bitstream_writing"] = data["time_measurements_bitstream_writing"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else sum(x)
            )
            data["time_measurements_gaussian_model"] = data["time_measurements_gaussian_model"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else sum(x)
            )
            
            # Compute average encoder times
            encoder_timings[seq].append([
                data["time_measurements_analysis"].mean(),
                data["time_measurements_hyper_analysis"].mean(),
                data["time_measurements_factorized_model_sender"].mean(),
                data["time_measurements_hyper_synthesis_sender"].mean(),
                data["time_measurements_gaussian_model"].mean(),
                data["time_measurements_geometry_comprresion"].mean(),
                data["time_measurements_bitstream_writing"].mean(),
            ])
            
            # Compute average decoder times
            decoder_timings[seq].append([
                data["time_measurements_bitstream_reading"].mean(),
                data["time_measurements_geometry_decompression"].mean(),
                data["time_measurements_factorized_model_receiver"].mean(),
                data["time_measurements_hyper_synthesis_receiver"].mean(),
                data["time_measurements_guassian_model"].mean(),
                data["time_measurements_synthesis_transform"].mean(),
            ])

    # Create DataFrames for encoder and decoder timings
    encoder_df = pd.DataFrame(encoder_timings, index=range(1, highest_fps + 1))
    decoder_df = pd.DataFrame(decoder_timings, index=range(1, highest_fps + 1))

    plot_stacked_bar_chart(encoder_df, 
        steps=["E1", "E2", "E3", "E4", "E5", "E6", "E7"], 
        labels=["E1: Analysis", "E2: Hyper Analysis", "E3: Factorized Bottleneck", "E4: Hyper Synthesis", "E5: Gaussian Bottleneck", "E6: G-PCC", "E7: Bitstream Wiriting"],
        title="encoder_timings_bars")
    plot_stacked_bar_chart(decoder_df, 
        steps=["D1", "D2", "D3", "D4", "D5", "D6"], 
        labels=["D1 - Bitstream Reading", "D2 - G-PCC", "D3 - Factorized Bottleneck", "D4 - Hyper Synthesis", "D5 - Gaussian Bottleneck", "D6 - Synthesis"],
        title="decoder_timing_bars")
 

def plot_coding_times_vs_num_points(all_data):
    # Initialize dictionaries to store aggregated data
    encoder_timings = {seq: [] for seq in sequences}
    decoder_timings = {seq: [] for seq in sequences}
    
    all_data["encoding_time"] = all_data["timestamps_codec_end_sender"] - all_data["timestamps_codec_start_sender"]
    all_data["decoding_time"] = all_data["timestamps_codec_end_receiver"] - all_data["timestamps_codec_start_receiver"]

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    fig2, ax2 = plt.subplots(figsize=(6, 3))

    cmap = cm.tab10  # You can change this to any other colormap like 'plasma', 'inferno', etc.
    norm = mcolors.Normalize(vmin=0.5, vmax=10.5)

    for seq in sequences:
        for fps in range(1, highest_fps + 1):
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            encoding_times = data["encoding_time"][3:]
            decoding_times = data["decoding_time"][3:]
            num_points = data["gop_info_num_points"][3:]

            color = cmap(norm(fps))
            
            ax1.scatter(num_points, encoding_times * 1000, color=[color] * len(num_points))
            ax2.scatter(num_points, decoding_times * 1000, color=[color] * len(num_points))

    # Add colorbars to indicate the FPS gradient
    fig1.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, ticks=range(1,11), label="Frames per Segment")
    fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2, ticks=range(1,11), label="Frames per Segment")

    # Set titles and labels
    ax1.set_xlabel("Number of Points")
    ax1.set_ylabel("Encoding Time (ms)")
    
    ax2.set_xlabel("Number of Points")
    ax2.set_ylabel("Decoding Time (ms)")

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(True)
    ax2.grid(True)    

    ax1.set_xticks([0, 25000, 50000, 75000, 100000, 125000, 150000])
    ax2.set_xticks([0, 25000, 50000, 75000, 100000, 125000, 150000])
            
    fig1.savefig(os.path.join(figures_path, "encoding_times.pdf"), bbox_inches="tight") 
    fig2.savefig(os.path.join(figures_path, "decoding_times.pdf"), bbox_inches="tight") 



def plot_stacked_bar_chart(df, steps, labels,  title):
    colors = {
        "E1": "#00FF81",  "E2": "#00FF81",  "E3": "#FFCCFF", "E4": "#00FF81", "E5": "#FFCCFF", "E6": "#FFCCFF", "E7": "#FFCCFF", 
        "D1": "#FFCCFF", "D2": "#FFCCFF", "D3": "#FFCCFF", "D4": "#00FF81", "D5": "#FFCCFF", "D6": "#00FF81"   
    }
    hatches = { 
        "E1": "//",  "E2": "XX",  "E3": "++", "E4": "..", "E5": "oo", "E6": "//", "E7": "XX", 
        "D1":"XX", "D2":"//", "D3":"++", "D4":"..", "D5":"..", "D6":"XX"   
    }
    bar_width = 0.8  # Width of each bar
    x = np.arange(1, highest_fps + 1)  # FPS values
        
    fig, ax = plt.subplots(figsize=(8, 4))
    offsets = np.arange(0, len(sequences) * bar_width, bar_width)  # Bar positions for each sequence

    # Iterate over sequences and plot each step
    for idx, seq in enumerate(sequences):
        bottom = np.zeros(len(x))  # Initialize bottom of the bars for stacking
        for step_idx, step in enumerate(steps):
            # Extract the relevant timing values for each FPS and sequence
            values = df[seq].apply(lambda x: x[step_idx] if isinstance(x, list) else 0) * 1000
            if idx == 0:
                ax.bar(x + offsets[idx], values, bar_width, bottom=bottom, hatch=hatches[step], edgecolor="black", color=colors[step], label=labels[step_idx])
            else:
                ax.bar(x + offsets[idx], values, bar_width, bottom=bottom, hatch=hatches[step], edgecolor="black", color=colors[step])
            bottom += values  # Update the bottom for the next step to stack the bars

    # Customizing the plot
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    ax.set_xlim([0.5, 10.5])
    #ax.plot([0.5, 10.5], [1000, 1000], color="gray", linestyle="dashed")

    ax.set_xlabel("Frames per segment")
    ax.set_ylabel("Average Time (ms)")
    ax.set_xticks(x + bar_width * (len(sequences) - 1) / 2)
    ax.set_xticklabels(range(1, highest_fps + 1))
    ax.legend(loc="upper left")#, bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.savefig(os.path.join(figures_path, f"{title}.pdf"), bbox_inches="tight") 


def plot_end_to_end_latency(all_data):
    for idx, seq in enumerate(sequences):
        for fps in range(1, 11):
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            succesfull_data = data.loc[(data["packet_received"] == True)]
            unsuccesfull_data = data.loc[data["packet_received"] == False]

            x = range(0, len(data))
            base_time = data["timestamps_capturing"].apply(lambda x: x[0] if isinstance(x, list) else eval(x)[0])
            playout_time = data["timestamps_playout"].apply(lambda x: eval(x)[0] if isinstance(x, str) else None)
            print(playout_time)

            s_1 =  data["timestamps_sampling"] - base_time
            s_2 =  data["timestamps_codec_end_sender"] - data["timestamps_sampling"]
            s_3 =  data["timestamps_server_published"] - data["timestamps_codec_end_sender"]

            r_1 =  data["timestamps_client_received"] - data["timestamps_server_published"]
            r_2 =  data["timestamps_codec_end_receiver"] - data["timestamps_client_received"]
            r_3 =  playout_time - data["timestamps_codec_end_receiver"]
            


            fig, ax = plt.subplots(figsize=(8, 4))
        
            labels = ["Capturer", "Enc. queue", "Encoder", "Publishing", "Transmission", "Decoder", "Buffer"]
            hatches = ["XX", "//" , "XX", "//", "XX", "//"]
            colors = ["#FFCCFF", "#00FF81", "red", "green", "#00FF81", "grey"]
            stacks = ax.stackplot(x, 
                s_1, s_2, s_3,
                r_1, r_2, r_3,
                colors=colors,
                labels=labels)

            plt.plot(x, sum([s_1, s_2, s_3]), label="Sender - Client", color="red")

            for stack, hatch in zip(stacks, hatches):
                stack.set_hatch(hatch)

            ax.set_xlabel("Segment ID")
            ax.set_ylabel("Latency from capturing (s)")
            ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
            ax.set_xlim(0, len(data) - 1)

            plt.savefig(os.path.join(figures_path, f"end-to-end-latency_{seq}_{fps}fps.pdf"), bbox_inches="tight") 
        




if __name__ == "__main__":
    plot()