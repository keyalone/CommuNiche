# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:09:00 2025

@author: lihs
"""

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from anndata import AnnData
from typing import Optional
import seaborn as sns
import palettes


def _assign_color(value, color: list):
    color_dict = dict()
    for i in range(len(value)):
        color_dict[value[i]] = color[i]
    return color_dict


def _convert_pval_to_asterisks(pval):
    if pval <= 0.0001:
        return "***"
    elif pval <= 0.001:
        return "**"
    elif pval <= 0.05:
        return "*"
    return ""


def _set_palette(length):

    if length <= 10:
        palette = palettes.default_10
    elif length <= 20:
        palette = palettes.default_20
    elif length <= 28:
        palette = palettes.default_28
    elif length <= 57:
        palette = palettes.default_57
    elif length <= len(palettes.default_102):  # 103 colors
        palette = palettes.default_102
    else:
        palette = ['grey' for _ in range(length)]
        print(
            'the obs value has more than 103 categories. Uniform '
            "'grey' color will be used for all categories."
        )

    return palette


def _melt_df(df: DataFrame, library_key: str, select_niche: Optional[list] = None, order: Optional[list] = None, ):
    if select_niche is not None:
        df = df[df['scNiche'].isin(select_niche)]
    if order is not None:
        df['scNiche'] = pd.Categorical(df['scNiche'], categories=order)
    df_melt = pd.melt(df, id_vars=[library_key, 'scNiche', 'Niche_ratio'])
    return df_melt


def stacked_barplot(adata: AnnData, x_axis: str, y_axis: str, mode: str = 'proportion', palette: Optional[list] = None,
                    save: bool = False, save_dir: str = '', kwargs: dict = {}):
    assert (mode.lower() in ['proportion', 'absolute']), 'mode should be either `proportion` or `absolute`!'

    length = len(adata.obs[y_axis].astype('category').cat.categories)
    if palette is None:
        palette = _set_palette(length=length)
    df = adata.obs.groupby([x_axis, y_axis]).size().unstack().fillna(0)
    if mode.lower() == 'proportion':
        df = df.div(df.sum(axis=1), axis=0)

    # plot
    ax = df.plot(kind='bar', stacked=True, width=0.75, color=palette, linewidth=0, **kwargs)
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
              ncol=(1 if length <= 14 else 2 if length <= 30 else 3), frameon=False)
    if save:
        plt.savefig(save_dir, format='svg')



from matplotlib.colors import ListedColormap

def enrichment_heatmap(cell_type_abundance,
                       pval_adjust,
                       binarized: bool = False,
                       show_pval: bool = False,
                       col_order: Optional[list] = None,
                       row_order: Optional[list] = None,
                       anno_key: Optional[str] = None,
                       anno_palette: Optional[list] = None,
                       save: bool = False,
                       save_dir: str = '',
                       kwargs: dict = {},
                       alpha: float = 0.05,
                       filter_nonsig: bool = True,   # 是否过滤非显著
                       xtick_rotation: int = 0       # 👈 新增参数（默认45°）
                       ):
    fc = cell_type_abundance.T.copy()
    pval = pval_adjust.T.copy()
    kwargs['vmin'] = 0

    # set order
    if col_order is not None:
        fc = fc[col_order]
        pval = pval[col_order]
        kwargs['col_cluster'] = False
    else:
        kwargs['col_cluster'] = True
    if row_order is not None:
        fc = fc.loc[row_order]
        pval = pval.loc[row_order]
        kwargs['row_cluster'] = False
    else:
        kwargs['row_cluster'] = True

    anno = fc.index
    length = len(anno.unique())
    if anno_palette is None:
        anno_palette = _set_palette(length=length)
    row_colors = dict(zip(anno.unique(), anno_palette))

    fc_plot = fc.copy()
    nonsig_mask = pval > alpha

    # 根据参数决定是否过滤
    if filter_nonsig:
        if binarized:
            fc_plot = fc_plot.applymap(lambda x: 0 if x <= 0 else 1)
            fc_plot = fc_plot.mask(nonsig_mask, -1e-9)
            kwargs['vmax'] = 1
            cmap = ListedColormap(['white', 'green'])
            cmap.set_under('white')
            cmap.set_bad('white')
            kwargs['cmap'] = cmap
        else:
            cmap_name = kwargs.get('cmap', 'magma')
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            cmap.set_under('white')
            cmap.set_bad('white')
            kwargs['cmap'] = cmap
            fc_plot = fc_plot.mask(nonsig_mask, -1e-9)
    else:
        # 不过滤
        if binarized:
            fc_plot = fc_plot.applymap(lambda x: 0 if x <= 0 else 1)
            kwargs['vmax'] = 1
            cmap = ListedColormap(['white', 'green'])
            kwargs['cmap'] = cmap
        else:
            cmap_name = kwargs.get('cmap', 'magma')
            cmap = sns.color_palette(cmap_name, as_cmap=True)
            kwargs['cmap'] = cmap

    # 显示显著性星号
    if show_pval:
        pval_str = pval.applymap(_convert_pval_to_asterisks)
        kwargs['annot'] = pval_str
        kwargs['fmt'] = ''

    # 绘制聚类热图
    ax = sns.clustermap(fc_plot,
                        method='complete',
                        row_colors=anno.map(row_colors),
                        **kwargs)

    # 图例
    for label, color in row_colors.items():
        ax.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    ax.ax_col_dendrogram.legend(
        bbox_to_anchor=(1, 0.5), loc='center left',
        ncol=(1 if length <= 14 else 2 if length <= 30 else 3),
        frameon=False
    )

    # 旋转坐标轴标签
    for tick in ax.ax_heatmap.get_yticklabels():
        tick.set_rotation(0)
    for tick in ax.ax_heatmap.get_xticklabels():
        tick.set_rotation(xtick_rotation)

    if save:
        plt.savefig(save_dir, bbox_inches='tight', dpi=300)

    return ax


