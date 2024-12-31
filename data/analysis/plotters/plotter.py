import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as ticker

CB_PALETTE = ["#332288","#117733","#44AA99","#88CCEE","#DDCC77","#CC6677","#AA4499","#882255"][::-1]
EXTRA_CB_PALLETTE = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

def get_pie_bar_zoom_in(data: np.array, labels: np.array, figsize=(12, 8), colors=None,
                        zoomed_in_y_axis_label="Percentage",
                        zoomed_in_x_axis_label="Values",
                        zoomed_in_pct_to_num = -1,
                        zoomed_in_fontsize = 8,
                        fig_title = "",
                        already_existing_figure=None):
    plt.ioff()  # disable interactive showing so that figures don't get shown upon creation

    if colors is None:
        colors = CB_PALETTE+EXTRA_CB_PALLETTE
    assert np.all(np.isclose(data.sum(), [100])), f"Sum of data is not 100 (%), but rather {data.sum()}"
    assert np.any(np.greater_equal(data, 80)), f"None of the elements are significant (>80%), {data}"
    assert len(colors) >= len(data)
    assert len(data) == len(labels)
    assert len(data) >= 3, "This kind of plot doesn't make sense for only two classes"
    sorted_idcs = np.argsort(data)[::-1]  # sort in descending order
    data = data[sorted_idcs]
    if type(labels) is list:
        labels = np.array(labels)
    labels = labels[sorted_idcs]

    fig = plt.figure(figsize=figsize) if already_existing_figure is None else already_existing_figure

    # Create axes for pie chart (bottom left) and bar chart (top right)
    ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.4])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.35, 0.5, 0.4, 0.4])

    # Pie chart (bottom left)
    explode = tuple([0.1] + [0]*(len(data)-1))

    def custom_pct(pct):
        idx = (np.abs(data - pct)).argmin() # nearest match
        label = labels[idx]
        string_formatted_pct = f"\n        {pct:.2f}%"
        # If the percentage of the value is smaller than 0.5, it will never be displayed correctly -- don't label it
        if pct < 0.5:
            return ""
        else:
            return ("   " if idx % 2 == 0 else ("\n" if idx > 2 else "")) + str(label) + (string_formatted_pct if idx == 0 else "")

    wedges, texts, autotexts = ax1.pie(data, explode=explode, colors=colors,
                                       autopct=custom_pct,
                                       pctdistance=0.8, labeldistance=1.05,
                                       startangle=30,
                                       counterclock=False)

    # Bar chart (top right)
    small_data = data[1:]  # Exclude the largest value
    x = range(len(small_data))
    bars = ax2.bar(x, small_data, color=colors[1:])
    ax2.set_ylim(0, max(small_data) * 1.2)
    ax2.set_ylabel(zoomed_in_y_axis_label)
    ax2.set_xlabel(zoomed_in_x_axis_label)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels[1:],fontsize=zoomed_in_fontsize)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}%' if zoomed_in_pct_to_num == -1 else f'{(height*zoomed_in_pct_to_num)//100}', ha='center', va='bottom')

    # Connect pie slices to bars
    for i, (wedge, bar) in enumerate(zip(wedges[1:], bars)):
        anglea, angleb = wedge.theta1, wedge.theta2
        center = wedge.center
        x = center[0] + wedge.r * np.cos(np.deg2rad((anglea + angleb) / 2))
        y = center[1] + wedge.r * np.sin(np.deg2rad((anglea + angleb) / 2))

        con = ConnectionPatch(xyA=(x, y), xyB=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                              coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color=colors[i + 1])
        fig.add_artist(con)

    # Titles
    ax1.set_title('Full Data Distribution', y=-0.05)
    ax2.set_title('Zoom-in on smaller values')
    fig.suptitle(fig_title,y=0)

    plt.ion()  # re-enable interactive showing
    plt.close()
    return fig


def get_time_graph(grouped_addresses_df,color_dict = None):
    fig,ax = plt.subplots(figsize=(15,10))
    assert len(grouped_addresses_df) == len(grouped_addresses_df["stacktrace"].unique())
    num_groups = len(grouped_addresses_df)
    ret_color_dict = {}
    if color_dict is None:
        cmap = plt.cm.jet(np.linspace(0,1,num_groups))
    for i,row in grouped_addresses_df.iterrows():
        if color_dict is None:
            ret_color_dict[row["stacktrace"]] = cmap[i]
        else: 
            assert row["stacktrace"] in color_dict, f"{row['stacktrace']} not in {color_dict.keys()}"
        
        ax.scatter(row["index_occurrences"],row["pages"],color=[(ret_color_dict if color_dict is None else color_dict)[row["stacktrace"]]]*len(row["pages"]),alpha=0.6, label=row["stacktrace"],marker='.')
    #fig.legend(title="Groups")
    ax.set_xlabel('Time')
    current_xlims = ax.get_xlim()
    ax.set_xlim([-10,current_xlims[-1]])
    ax.set_ylabel('Page faulted (hex)')
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x,pos: hex(int(x))))
    return fig,ax,ret_color_dict

def get_regs_graph():
    pass