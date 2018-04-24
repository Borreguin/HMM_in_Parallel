import numpy as np
import matplotlib.pyplot as plt


def set_iCol_and_iRow(iCol, iRow, nrows, ncols):
    iCol += 1
    if iRow >= nrows:
        iRow = 0
    if iCol >= ncols:
        iCol = 0
        iRow += 1
    return iCol, iRow

def plot_profiles_24h(df_x, df_y, states_to_plot, yLim, units, n_col=4, figsize=(16,24)):
    nrows = int(np.ceil(len(states_to_plot)/n_col))
    n_samples = len(df_x.loc[df_x.index[0]])
    if n_samples == 24:
        labels = range(24)
    elif n_samples > 24:
        fc = n_samples/24
        labels =[x/fc for x in range(int(fc*n_samples))]
    else:
        labels = range(n_samples)
    #print(nrows)
    if nrows > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=n_col, figsize=figsize)
        medianprops = dict(linewidth=4, color='red')
        i, j = 0, -1
        for n in states_to_plot:
            mask = df_y[df_y['hidden_states'] == n].index
            j, i = set_iCol_and_iRow(j, i, nrows, n_col)
            if(len(mask))>0:
                df_to_plot = df_x.loc[mask]
                df_to_plot.plot.box(ax=axes[i][j],notch=False,  medianprops=medianprops, showfliers=True)
                axes[i][j].set_ylim(yLim)
                axes[i][j].set_xlabel('Hours')
                #axes[-1][j].set_xlabel('Hours')
                axes[i][0].set_ylabel('[ ' +  units + ' ]')
                axes[i][j].set_title('ID_= ' + str(n) + ' #Days=' + str(len(mask)))
                axes[i][j].set_xticklabels(labels= labels, rotation=-90)
                for label in axes[i][j].get_xticklabels()[::2]:
                    label.set_visible(False)
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=n_col, figsize=figsize)
        medianprops = dict(linewidth=4, color='red')
        i, j = 0, -1
        for n in states_to_plot:
            mask = df_y[df_y['hidden_states'] == n].index
            j, i = set_iCol_and_iRow(j, i, nrows, n_col)
            if(len(mask))>0:
                df_to_plot = df_x.loc[mask]
                df_to_plot.plot.box(ax=axes[j],notch=False,  medianprops=medianprops, showfliers=True)
                axes[j].set_ylim(yLim)
                axes[j].set_xlabel('Hours')
                #axes[-1][j].set_xlabel('Hours')
                axes[j].set_ylabel('[ ' +  units + ' ]')
                axes[j].set_title('ID_= ' + str(n) + ' #Days=' + str(len(mask)))
                axes[j].set_xticklabels(labels = labels, rotation=-90)
    plt.tight_layout()
    plt.show()