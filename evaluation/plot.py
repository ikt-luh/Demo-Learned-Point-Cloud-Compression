import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
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
    "#2e0ff5", "#007295", "#008aa7", "#00a6bb", "#00c9b7", "#00ed94", 
    "#bcff8a", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb"
]

# Templates
logs_base_path = "./logs"
figures_path = "./figures"
sequences = ["final_test"]
fps_min = 1
fps_max = 8


def plot():
    data = load_data()

    # Encoder and Decoder Latency
    plot_coding_times(data)
    plot_coding_times_vs_num_points(data)
    plot_end_to_end_latency(data)
    plot_actual_times(data)
# Load the data

def load_data(save=False):
    merged_results = []

    for fps in range(fps_min, fps_max+1):
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
    encoder_actual_time = {seq: [] for seq in sequences}
    decoder_actual_time = {seq: [] for seq in sequences}
    
    for seq in sequences:
        for fps in range(fps_min, fps_max + 1):
            # Filter data for the specific sequence and FPS
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            
            # Preprocessing
            data["time_measurements_bitstream_writing"] = data["time_measurements_bitstream_writing"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else x 
            )
            data["time_measurements_gaussian_model"] = data["time_measurements_gaussian_model"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else x
            )
            
            # Compute average encoder times
            encoder_timings[seq].append([
                data["time_measurements_analysis"].mean(),
                data["time_measurements_hyper_analysis"].mean(),
                data["time_measurements_factorized_model_sender"].mean(),
                data["time_measurements_hyper_synthesis_sender"].mean(),
                data["time_measurements_gaussian_model"].mean(),
                data["time_measurements_geometry_compression"].mean(),
                data["time_measurements_bitstream_writing"].mean(),
            ])
            encoder_actual_time[seq].append(data["timestamps_codec_end_sender"] - data["timestamps_codec_start_sender"])
            
            # Compute average decoder times
            decoder_timings[seq].append([
                data["time_measurements_bitstream_reading"].mean(),
                data["time_measurements_geometry_decompression"].mean(),
                data["time_measurements_factorized_model_receiver"].mean(),
                data["time_measurements_hyper_synthesis_receiver"].mean(),
                data["time_measurements_guassian_model"].mean(),
                data["time_measurements_synthesis_transform"].mean(),
                
            ])
            decoder_actual_time[seq].append(data["timestamps_codec_end_receiver"] - data["timestamps_codec_start_receiver"])

    # Create DataFrames for encoder and decoder timings
    encoder_df = pd.DataFrame(encoder_timings, index=range(fps_min, fps_max + 1))
    decoder_df = pd.DataFrame(decoder_timings, index=range(fps_min, fps_max + 1))

    plot_stacked_bar_chart(encoder_df, encoder_actual_time,
        steps=["E1", "E2", "E3", "E4", "E5", "E6", "E7"], 
        labels=["E1: Analysis Transform", "E2: Hyper Analysis Transform", "E3: Factorized Bottleneck", "E4: Hyper Synthesis Transform", "E5: Gaussian Bottleneck", "E6: G-PCC", "E7: Bitstream Writing", "Coding Total"],
        title="encoder_timings_bars")
    plot_stacked_bar_chart(decoder_df, decoder_actual_time,
        steps=["D1", "D2", "D3", "D4", "D5", "D6"], 
        labels=["D1: Bitstream Reading", "D2: G-PCC", "D3: Factorized Bottleneck", "D4: Hyper Synthesis Transform", "D5: Gaussian Bottleneck", "D6: Synthesis Transform", "Coding Total"],
        title="decoder_timing_bars")
 

def plot_coding_times_vs_num_points(all_data):
    # Initialize dictionaries to store aggregated data
    encoder_timings = {seq: [] for seq in sequences}
    decoder_timings = {seq: [] for seq in sequences}
    
    all_data["encoding_time"] = all_data["timestamps_codec_end_sender"] - all_data["timestamps_codec_start_sender"]
    all_data["decoding_time"] = all_data["timestamps_codec_end_receiver"] - all_data["timestamps_codec_start_receiver"]

    fig1, ax1 = plt.subplots(figsize=(5, 2))
    fig2, ax2 = plt.subplots(figsize=(5, 2))
    
    fps_range = fps_max - fps_min + 1
    # Create a colormap
    colors = cm.viridis(np.linspace(0, 1, fps_range))  # Adjust to any colormap you prefer
    cmap = ListedColormap(colors)

    # Define bounds and norm for the colorbar
    bounds = np.arange(fps_min, fps_max + 2)  # Define boundaries for discrete segments
    norm = BoundaryNorm(bounds, cmap.N)

    for seq in sequences:
        for fps in range(fps_min, fps_max + 1):
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            encoding_times = data["encoding_time"][3:-1]
            decoding_times = data["decoding_time"][3:-1]
            num_points = data["gop_info_num_points"][3:-1]

            color = cmap(norm(fps))
            
            ax1.scatter(num_points, encoding_times * 1000, color=[color] * len(num_points), s= [5] * len(num_points))
            ax2.scatter(num_points, decoding_times * 1000, color=[color] * len(num_points), s= [5] * len(num_points))

    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=fps_min, vmax=fps_max))
    sm.set_array([])

    bin_centers = 0.5 * (bounds[:-1] + bounds[1:])
    cb1 = fig1.colorbar(sm, ax=ax1, ticks=bin_centers, boundaries = bounds, label="Frames per Segment")
    cb2 = fig2.colorbar(sm, ax=ax2, ticks=bin_centers, boundaries = bounds, label="Frames per Segment")
    cb1.ax.set_yticklabels(range(fps_min, fps_max + 1))
    cb2.ax.set_yticklabels(range(fps_min, fps_max + 1))
    #cb1.set_xlim([fps_min-0.5, fps_max+0.5])
    #cb2.set_xlim([fps_min-0.5, fps_max+0.5])
    
    ax1.annotate( "Efficient encoding", 
        fontsize=10, xy=(50000, 250), rotation=16         # End position color='black'
    )
    ax1.annotate(text="",xytext=(25000, 250), xy=(100000, 850), arrowprops=dict(arrowstyle="->"))
    ax1.annotate( "Resources\nexhausted", 
        fontsize=10, xy=(115000, 800), rotation=55        # End position color='black'
    )
    ax1.annotate(text="",xytext=(110000, 900), xy=(140000, 2000), arrowprops=dict(arrowstyle="->"))   
    
    ax2.annotate( "Efficient decoding", 
        fontsize=10, xy=(60000, 100), rotation=13         # End position color='black'
    )
    ax2.annotate(text="",xytext=(25000, 100), xy=(125000, 1150), arrowprops=dict(arrowstyle="->"))
    ax2.annotate( "Resources\nexhausted", 
        fontsize=10, xy=(128000, 850), rotation=74        # End position color='black'
    )
    ax2.annotate(text="",xytext=(130000, 1200), xy=(140000, 3000), arrowprops=dict(arrowstyle="->"))

    # Set titles and labels
    ax1.set_xlabel("Number of Points")
    ax1.set_ylabel("Encoding Time (ms)")
    
    ax2.set_xlabel("Number of Points")
    ax2.set_ylabel("Decoding Time (ms)")

    # Arrows 



    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(True)
    ax2.grid(True)    

    ax1.set_xticks([0,  50000,  100000,  150000])
    ax2.set_xticks([0,  50000,  100000,  150000])
            
    fig1.savefig(os.path.join(figures_path, "encoding_times.pdf"), bbox_inches="tight") 
    fig2.savefig(os.path.join(figures_path, "decoding_times.pdf"), bbox_inches="tight") 



def plot_stacked_bar_chart(df, actual_times, steps, labels,  title):
    col_seq = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4']
    colors = {
        "E1": col_seq[0],  "E2": col_seq[1],  "E3": col_seq[2], "E4": col_seq[4], "E5": col_seq[5], "E6": col_seq[7], "E7": col_seq[8], 
        "D1": col_seq[8], "D2": col_seq[7], "D3": col_seq[2], "D4": col_seq[3], "D5": col_seq[4], "D6": col_seq[0]
    }
    hatches = { 
        "E1": "//",  "E2": "XX",  "E3": "++", "E4": "..", "E5": "oo", "E6": "//", "E7": "XX", 
        "D1":"XX", "D2":"//", "D3":"++", "D4":"..", "D5":"..", "D6":"XX"   
    }
    bar_width = 0.8  # Width of each bar
    x = np.arange(fps_min, fps_max + 1)  # FPS values
        
    fig, ax = plt.subplots(figsize=(8, 4))
    offsets = np.arange(0, len(sequences) * bar_width, bar_width)  # Bar positions for each sequence

    # Iterate over sequences and plot each step
    for idx, seq in enumerate(sequences):
        bottom = np.zeros(len(x))  # Initialize bottom of the bars for stacking
        for step_idx, step in enumerate(steps):
            # Extract the relevant timing values for each FPS and sequence
            values = df[seq].apply(lambda x: x[step_idx] if isinstance(x, list) else 0) * 1000
            if idx == 0:
                ax.bar(x + offsets[idx], values, bar_width, bottom=bottom, 
                #hatch=hatches[step], 
                edgecolor="black", color=colors[step], label=labels[step_idx])
            else:
                ax.bar(x + offsets[idx], values, bar_width, bottom=bottom, 
                #hatch=hatches[step], 
                edgecolor="black", color=colors[step])
            bottom += values  # Update the bottom for the next step to stack the bars

        # ERROR BARS (optional)
        """
        time_means = [v.mean() * 1000 for v in actual_times[seq]]
        time_stddev = [v.std() * 1000 for v in actual_times[seq]]
        err_low = [m - std for m, std in zip(time_means, time_stddev)]
        err_high = [m + std for m, std in zip(time_means, time_stddev)]
        print(err_low[-1] - err_high[-1])
        ax.errorbar(x, time_means, yerr=time_stddev,
            label=labels[-1],
            color="black",
            marker="o",
            capsize=5
        )
        """

    # Customizing the plot
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    ax.set_xlim([fps_min-0.5, fps_max + 0.5])
    #ax.plot([0.5, 10.5], [1000, 1000], color="gray", linestyle="dashed")

    ax.set_xlabel("Frames per segment")
    ax.set_ylabel("Average Time (ms)")
    ax.set_xticks(x + bar_width * (len(sequences) - 1) / 2)
    ax.set_xticklabels(range(fps_min, fps_max + 1))
    ax.legend(loc="upper left")#, bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.savefig(os.path.join(figures_path, f"{title}.pdf"), bbox_inches="tight") 


def plot_end_to_end_latency(all_data):
    for idx, seq in enumerate(sequences):
        for fps in range(fps_min, fps_max + 1):
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)][3:]
            succesfull_data = data.loc[(data["packet_received"] == True)]
            unsuccesfull_data = data.loc[data["packet_received"] == False]

            x = range(0, len(data))
            base_time = data["timestamps_capturing"].apply(lambda x: x[0] if isinstance(x, list) else eval(x)[0])
            playout_time = data["timestamps_playout"].apply(lambda x: eval(x)[0] if isinstance(x, str) else None)

            s_1 =  data["timestamps_sampling"] - base_time
            s_2 =  data["timestamps_codec_end_sender"] - data["timestamps_sampling"]
            s_3 =  data["timestamps_server_published"] - data["timestamps_codec_end_sender"]

            r_1 =  data["timestamps_client_received"] - data["timestamps_server_published"]
            r_2 =  data["timestamps_codec_end_receiver"] - data["timestamps_client_received"]
            r_3 =  playout_time - data["timestamps_codec_end_receiver"]
            


            fig, ax = plt.subplots(figsize=(8, 3))
        
            labels = ["Capturer", "Encoder", "Publishing", "Transmission", "Decoder", "Buffer"]
            hatches = ["XX", "//" , "XX", "//", "XX", "//"]
            colors = ["#ffcdad", "#f66909", "#522303", "#add8fc", "#088bf7", "#032e52"]
            stacks = ax.stackplot(x, 
                s_1, s_2, s_3,
                r_1, r_2, r_3,
                colors=colors,
                labels=labels)

            #plt.plot(x, sum([s_1, s_2, s_3]), label="Sender - Client", color="red")

            """
            for stack, hatch in zip(stacks, hatches):
                stack.set_hatch(hatch)
            """

            ax.set_xlabel("Segment ID")
            ax.set_ylabel("Latency from capturing (s)")
            #ax.legend(loc="center left", bbox_to_anchor=(1,0.5))
            ax.legend(loc=9, bbox_to_anchor=(0.5, -0.17), ncol=3)
            ax.set_xlim(0, len(data) - 1)

            plt.savefig(os.path.join(figures_path, f"end-to-end-latency_{seq}_{fps}fps.pdf"), bbox_inches="tight") 
        

def plot_actual_times(all_data):
       # Initialize dictionaries to store aggregated data
    encoder_timings = {seq: [] for seq in sequences}
    decoder_timings = {seq: [] for seq in sequences}
    encoder_actual_time = {seq: [] for seq in sequences}
    decoder_actual_time = {seq: [] for seq in sequences}
    
    for seq in sequences:
        for fps in range(fps_min, fps_max + 1):
            # Filter data for the specific sequence and FPS
            data = all_data.loc[(all_data["fps_sender"] == fps) & (all_data["sequence_sender"] == seq)]
            
            # Preprocessing
            data["time_measurements_bitstream_writing"] = data["time_measurements_bitstream_writing"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else x 
            )
            data["time_measurements_gaussian_model"] = data["time_measurements_gaussian_model"].apply(
                lambda x: sum(eval(x)) if isinstance(x, str) else x
            )
            
            # Compute average encoder times
            encoder_times = [
                data["time_measurements_analysis"],
                data["time_measurements_hyper_analysis"],
                data["time_measurements_factorized_model_sender"],
                data["time_measurements_hyper_synthesis_sender"],
                data["time_measurements_gaussian_model"],
                data["time_measurements_geometry_compression"],
                data["time_measurements_bitstream_writing"],
            ]
            encoder_timings[seq].append(sum(encoder_times))
            encoder_actual_time[seq].append(data["timestamps_codec_end_sender"] - data["timestamps_codec_start_sender"])
            
            # Compute average decoder times
            decoder_times = [
                data["time_measurements_bitstream_reading"],
                data["time_measurements_geometry_decompression"],
                data["time_measurements_factorized_model_receiver"],
                data["time_measurements_hyper_synthesis_receiver"],
                data["time_measurements_guassian_model"],
                data["time_measurements_synthesis_transform"],
            ]
            decoder_timings[seq].append(sum(decoder_times))
            decoder_actual_time[seq].append(data["timestamps_codec_end_receiver"] - data["timestamps_codec_start_receiver"])

    for idx, seq in enumerate(sequences):
        fig, ax = plt.subplots(figsize=(5,2))

        enc_time_diff = [(act - comp)/ act for comp, act in zip(encoder_timings[seq],encoder_actual_time[seq])]
        enc_mean = [np.mean(t) for t in enc_time_diff]
        enc_stddev = [np.std(t) for t in enc_time_diff]

        dec_time_diff = [(act - comp)/ act  for comp, act in zip(decoder_timings[seq],decoder_actual_time[seq])]
        dec_mean = [np.mean(t) for t in dec_time_diff]
        dec_stddev = [np.std(t) for t in dec_time_diff]
        
        x = range(0, len(dec_mean))
        ax.errorbar(x, enc_mean, yerr=enc_stddev, capsize=5, label="Encoding")
        ax.errorbar(x, dec_mean, yerr=dec_stddev, capsize=5, label="Decoding")
        ax.yaxis.grid(color='gray')
        ax.set_xlabel("Frames per Segment")
        ax.legend()

        fig.savefig(os.path.join(figures_path, f"time-comp-vs-actual_{seq}.pdf"), bbox_inches="tight") 


    

if __name__ == "__main__":
    plot()