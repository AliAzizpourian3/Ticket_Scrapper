color_dic={
    "default" : ["#1E90FF", "#50C878", "#9400D3",  "#E97451", "#DC143C", "#006B3C", "#9B111E", "#0047AB", "#36454F", "#DA70D6", "#FFBF00"],
    "big": [ "#FF0000", "#AB3428", "#DE995D", "#FFA500", "#FFD700", "#00FF00", "#008000",  "#00FFFF", "#87CEEB", "#4169E1", "#000080", "#4B0082", "#DDA0DD", "#FF00FF", "#FFC0CB", "#FFB6C1", "#D2B48C", "#8B4513", "#36454F",  "#000000", "#FFD700", "#C0C0C0", "#B22222"],
    "saturated" : ['#00ACC1', '#FFA07A',  '#00C853', '#AA00FF','#FFCA28', '#2979FF', '#FF4081', '#E0115F'],
    "earth" : ['#F4A460', '#6B8E23', '#E97451', '#C19A6B', '#E0B0FF','#635147', '#704214', '#ACE1AF', '#AF8751', '#D2691E',  '#FF7518', '#808000', '#483C32', '#8A3324', '#CD7F32'],
    "spring" : ['#FFB7C5', '#8DB600',  '#00CCCC', '#C8A2C8', '#FF7F50', '#Ffc0A1', '#967BB6'],
    "tropical" : ['#0077BE', '#32CD32', '#FFDA03', '#FD7C6E', '#DA70D6', '#87CEEB', '#FC8EAC', '#93E9BE', '#FF8C00'],
    "mega big": [ "#FF0000", "#AB3428", "#DE995D", "#FFA500", "#FFD700", "#00FF00", "#008000", "#00FFFF", "#87CEEB", "#4169E1", "#FF4500", "#DDA0DD", "#FF00FF", "#FFC0CB", "#FFB6C1", "#D2B48C", "#8B4513", "#A52A2A", "#FF6347", "#C0C0C0", "#B22222", "#FA8072", "#DC143C", "#00BFFF", "#228B22", "#FFDAB9", "#6A5ACD", "#FF6347", "#ADFF2F", "#DA70D6", "#32CD32", "#BA55D3", "#7B68EE", "#20B2AA", "#B0C4DE", "#D2691E", "#F08080", "#8FBC8F", "#FFA07A", "#00FA9A", "#F0E68C", "#8A2BE2", "#5F9EA0", "#CD5C5C", "#FF1493", "#FFD700", "#C71585", "#40E0D0", "#FF69B4", "#FF4500", "#8B0000" ],
    "mega big2": [ "#FF0000", "#AB3428", "#DE995D", "#FFA500", "#008000", "#87CEEB", "#4169E1", "#DDA0DD", "#FF00FF", "#FFC0CB", "#D2B48C", "#8B4513", "#FF6347", "#C0C0C0", "#B22222", "#DC143C", "#00BFFF", "#228B22", "#FFDAB9", "#6A5ACD", "#FF6347", "#ADFF2F", "#DA70D6", "#32CD32", "#BA55D3", "#7B68EE", "#20B2AA", "#B0C4DE", "#D2691E", "#F08080", "#8FBC8F", "#FFA07A", "#00FA9A", "#F0E68C", "#8A2BE2", "#5F9EA0", "#CD5C5C", "#FF1493", "#FFD700", "#C71585", "#40E0D0", "#FF69B4", "#FF4500", "#8B0000" ],
    "paper" :  ['#3B82F6', '#34D399', '#DC2626', '#F056A1', '#FB923C', '#9B4D96', '#FBBF24', '#AD6F3B'],
    "paper4" :  ['#3B82F6', '#34D399', '#F056A1', '#FB923C'],
    "jyuu_cool" : ["#ff595e","#ff924c","#ffca3a","#c5ca30","#8ac926","#52a675","#1982c4","#4267ac","#6a4c93","#d677b8"],
    "siemens_colors": ["#000028", "#CCD2D8", "#333353", "#00646e", "#c5c5b8", "#66667e", "#9999a9", "#ccccd4", "#e5e5e9", "#aaaa96", "#dfdfd9", "#f3f3f0", "#00af8e", "#00d7a0", "#009999", "#01ffb9", "#01e6dc", "#00C1B6", "#00F2D7", "#00FDBA", "#8a8a7a", "#f7c600", "#ffd732", "#ffe270","#00557c", "#553ba3", "#805cff", "#b4a8ff", "#ef0137", "#ec6602", "#ff9000", "#0087be",  "#00bedc"  ],
    "siemens_colors_graphs": ["#00af8e", "#0087be", "#805cff", "#01ffb9", "#ec6602", "#f7c600", "#ef0137", "#c5c5b8", "#333353", "#66667e", "#9999a9", "#ccccd4", "#e5e5e9", "#aaaa96", "#dfdfd9", "#f3f3f0",  "#00d7a0", "#009999", "#01e6dc", "#00C1B6", "#00F2D7", "#00FDBA", "#8a8a7a", "#ffd732", "#ffe270","#00557c", "#553ba3", "#b4a8ff", "#ff9000",  "#00bedc"  ],
    "scg": ["#00af8e", "#0087be", "#805cff", "#01ffb9", "#ec6602", "#f7c600", "#ef0137", "#f3f3f0", "#8f85d1", "#ff9000", "#00557c", "#ffe270", "#6E3710", "#F79EAD", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513", "#8B4513" ]


}

def prepare_subplots(axs, key="default", colors=[], background_color="w", tick_color="k", tick_linewidth=1.8, spine_linewidth=1.8, grid_color="gray", grid_linestyle_major="--", grid_linestyle_minor=":",  labelticksize=14, GRID=True, MINOR=True):

    import matplotlib.pyplot as plot
    import matplotlib as mpl
    #plot.rcParams["axes.prop_cycle"] = cycler(color=Color_dic["default"])
    #mpl.rcParams['axes.prop_cycle'] = plot.cycler(color=["red","brown","black"])
    if colors:
        new_color_cycle = plot.cycler(color=colors)
    else:
        new_color_cycle = plot.cycler(color=color_dic[key])
    for ax in axs:
        ax.set_prop_cycle(new_color_cycle)
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_edgecolor(tick_color)
        ax.xaxis.label.set_color(tick_color)
        ax.yaxis.label.set_color(tick_color)
        ax.title.set_color(tick_color)
        ax.tick_params(axis='both', width=tick_linewidth, labelsize=labelticksize, color=tick_color, labelcolor=tick_color)
        if GRID:
            ax.grid(True, color=grid_color,linestyle=grid_linestyle_major, linewidth=0.5,alpha=0.7, which='major')
            ax.grid(True, color=grid_color,linestyle=grid_linestyle_minor, linewidth=0.5,alpha=0.5, which='minor')
            ax.set_axisbelow(True)
        if MINOR:
            ax.minorticks_on()
        

def prepare_subplots_big(axs, key="default", colors=[], background_color="w", tick_color="k", tick_linewidth=1.5, spine_linewidth=1.5, grid_color="gray", grid_linestyle_major=((0,(3,5))), grid_linestyle_minor=":", labelticksize=16, GRID=True, MINOR=False):

    import matplotlib.pyplot as plot
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams["font.size"] = 11
    if colors:
        new_color_cycle = plot.cycler(color=colors)
    else:
        new_color_cycle = plot.cycler(color=color_dic[key])

    for ax in axs:
        ax.set_prop_cycle(new_color_cycle)
        ax.set_facecolor(background_color)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_linewidth)
            spine.set_edgecolor(tick_color)
        ax.xaxis.label.set_color(tick_color)
        ax.yaxis.label.set_color(tick_color)
        ax.title.set_color(tick_color)
        ax.tick_params(axis='both', width=tick_linewidth, labelsize=labelticksize,length=5,color=tick_color,labelcolor=tick_color )
        if GRID:
            ax.grid(True, color=grid_color,linestyle=grid_linestyle_major, linewidth=0.8,alpha=0.7, which='major')
            ax.grid(False, color=grid_color,linestyle=grid_linestyle_minor, linewidth=0.6,alpha=0.5, which='minor')
            # ax.grid(False, which="minor")
            ax.set_axisbelow(True)
        # ax.minorticks_on()
        # ax.tick_params(which='minor',axis='y', width=1.5,length=3.1)
        if MINOR:
            ax.tick_params(which='minor',axis='both', width=1.,length=3.,color=tick_color,labelcolor=tick_color)
            ax.minorticks_on()

def get_subplots(*args, key="default", colors=[], background_color="w", tick_color="k", tick_linewidth=1.8, spine_linewidth=1.8, grid_color="gray", grid_linestyle_major="--", grid_linestyle_minor=":", labelticksize=14, GRID=True, MINOR=True, **kwargs):
    import matplotlib.pyplot as plot
    import numpy as np
    fig, axs = plot.subplots(*args, **kwargs)
    if not isinstance(axs, (list, np.ndarray)):
        axs = np.array([axs])
    shape = axs.shape
    if len(shape) == 2:
        axs = axs.reshape(shape[0]*shape[1], )
    prepare_subplots(axs, key=key, colors=colors, background_color=background_color, tick_color=tick_color, tick_linewidth=tick_linewidth, spine_linewidth=spine_linewidth, grid_color=grid_color, grid_linestyle_major=grid_linestyle_major, grid_linestyle_minor=grid_linestyle_minor, labelticksize=labelticksize, GRID=GRID, MINOR=MINOR)
    if len(axs) == 1:
        axs = axs[0]
    if len(shape) == 2:
        axs = axs.reshape(shape)
    return fig, axs

def fancy_histogram(fig, ax, array, bins, weights=None, color = color_dic['default'][0],linestyle="-", linewidth = 2, alpha=0.1,label=None):
    import numpy as np
    if weights is None:
        weights = np.full(len(array),100)/(1e-12+len(array))
    # ax.hist(array, bins=bins, color=color, alpha=alpha, weights=weights,edgecolor=color)
    counts, bins,_ = ax.hist(array, bins=bins, color=color, weights=weights, histtype="step",linestyle=linestyle,linewidth=linewidth, label=label)
    bins = [bins[0]] +  [ element for i in range(1, len(bins)-1) for element in [bins[i], bins[i]]  ] + [bins[-1]]
    counts = [element for i in range(len(counts)) for element in [counts[i], counts[i]]]
    ax.fill_between(bins, [0]*len(bins), counts, alpha=alpha, color=color)


def arrange_twin_plots(ax,ax_twin,n_decimal_points=2):

    alpha, beta = ax.get_ylim()
    ymin, ymax = ax_twin.get_ylim()
    m = (beta-alpha)/(ymax-ymin)
    b = (alpha*ymax-beta*ymin)/(ymax-ymin)
    

    children = [ch for ch in ax_twin.get_children() if "Line2D" in str(ch) or "PathCollection" in str(ch)]

    ax_twin.set_yticks([y for y in ax.get_yticks()])
    ax_twin.set_ylim(alpha,beta)

    ax_twin.set_yticklabels(["{val:.{dp}f}".format(dp=n_decimal_points, val=(y-b)/(m)) for y in ax_twin.get_yticks()])



    for ch in children:
        # print(ch.get_offsets())
        if "Line2D" in str(ch):
            y_data = ch.get_ydata()
            y_data = m*y_data+b
            ch.set_ydata(y_data)
        elif "PathCollection" in str(ch):
            data = ch.get_offsets()
            data[:,1] = m*data[:,1]+b
            ch.set_offsets(data)

def arrange_bar_plots(ax):
    patches = sorted(ax.patches, key=lambda patch: patch.get_height(), reverse=True)
    xslim = ax.get_xlim()
    yslim = ax.get_ylim()
    xslabel = ax.get_xlabel()
    yslabel = ax.get_ylabel()
    ax.cla()
    for patch in patches:
        ax.add_patch(patch)
    
    ax.set_xlim(xslim)
    ax.set_ylim(yslim)
    ax.set_xlabel(xslabel)
    ax.set_ylabel(yslabel)
    return ax