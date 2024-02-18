import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_topics_3d_interactive(model_components, top_words_per_topic, output_filename):
    # Perform PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    reduced_components = pca.fit_transform(model_components)

    fig = go.Figure()

    # Updated color palette with sophisticated colors
    colors = [
        'rgba(10, 132, 255, opacity)',  # Vivid Blue
        'rgba(255, 55, 95, opacity)',  # Vibrant Red
        'rgba(52, 199, 89, opacity)',  # Bright Green
        'rgba(171, 0, 255, opacity)',  # Electric Purple
        'rgba(255, 94, 0, opacity)',  # Sunset Orange
        'rgba(153, 0, 153, opacity)',  # Deep Magenta
        'rgba(0, 206, 209, opacity)',  # Turquoise Blue
        'rgba(255, 127, 80, opacity)',  # Coral Pink
        'rgba(255, 247, 0, opacity)',  # Lemon Yellow
        'rgba(25, 25, 112, opacity)',  # Midnight Blue
        'rgba(255, 0, 56, opacity)',  # Cherry Red
        'rgba(181, 126, 220, opacity)',  # Lavender Purple
        'rgba(0, 128, 128, opacity)',  # Teal Green
        'rgba(255, 204, 153, opacity)',  # Peachy Pink
        'rgba(75, 0, 130, opacity)',  # Indigo Blue
        'rgba(255, 213, 79, opacity)',  # Mustard Yellow
        'rgba(224, 17, 95, opacity)',  # Ruby Red
        'rgba(135, 206, 235, opacity)',  # Sky Blue
        'rgba(152, 255, 152, opacity)',  # Mint Green
        'rgba(218, 112, 214, opacity)',  # Orchid Purple
        'rgba(0, 255, 255, opacity)',  # Cyan
        'rgba(255, 165, 0, opacity)',  # Orange
        'rgba(255, 99, 71, opacity)',  # Tomato
        'rgba(255, 20, 147, opacity)',  # DeepPink
        'rgba(255, 0, 255, opacity)',  # Magenta
        'rgba(255, 105, 180, opacity)',  # Pink
        'rgba(238, 130, 238, opacity)',  # Violet
        'rgba(255, 140, 0, opacity)',  # DarkOrange
        'rgba(255, 215, 0, opacity)',  # Gold
        'rgba(0, 128, 0, opacity)',  # Green
        'rgba(255, 192, 203, opacity)',  # Pink
        # Add more colors as needed, replacing 'opacity' with actual opacity values later
    ]

    for i, component in enumerate(reduced_components):
        top_word_info = top_words_per_topic[i]

        # Calculate opacity to enhance visual differentiation; adjust as needed
        opacity = 0.6 + 0.4 * (i / len(reduced_components))  # Gradually increasing opacity
        color = colors[i % len(colors)].replace('opacity', str(opacity))
        legend_group = f"topic_{i + 1}"

        # Main topic point without text
        fig.add_trace(go.Scatter3d(x=[component[0]], y=[component[1]], z=[component[2]],
                                   mode='markers',
                                   marker=dict(size=12, color=color, line=dict(width=2, color='DarkSlateGrey')),
                                   name=f'Topic {i + 1}',
                                   legendgroup=legend_group,
                                   hoverinfo='text'))

        # Calculate z-axis label offset for distancing the label higher on the x-axis
        z_label_offset = .3  # Adjust this value based on your dataset scale and visual preference
        label_z_position = component[2] + z_label_offset

        # Adding text trace for the label, positioned higher on the x-axis
        fig.add_trace(go.Scatter3d(x=[component[0]], y=[component[1]], z=[label_z_position],
                                   mode='text',
                                   text=f"Topic {i + 1} ",
                                   hoverinfo='none',
                                   legendgroup=legend_group,
                                   showlegend=False,
                                   textfont=dict(size=13, color="white")))

        # Word clusters with refined marker aesthetics
        for word, weight in top_word_info:
            word_position = component + np.random.normal(loc=0.0, scale=0.05, size=3)
            fig.add_trace(go.Scatter3d(x=[word_position[0]], y=[word_position[1]], z=[word_position[2]],
                                       mode='markers',
                                       marker=dict(size=8, color=color, line=dict(width=1, color='DarkSlateGrey')),
                                       text=f"{word} ({weight:.2f})",
                                       hoverinfo='text',
                                       legendgroup=legend_group,
                                       showlegend=False))

    # Refined layout for a sleek appearance
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3',
        xaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray", ),
        yaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray"),
        zaxis=dict(backgroundcolor="rgb(10, 10, 10)",
                   gridcolor="gray",
                   showbackground=True,
                   zerolinecolor="gray"),
    ),
                      paper_bgcolor="rgb(10, 10, 10)",
                      plot_bgcolor='rgb(10, 10, 10)',
                      font=dict(color="white"),
                      legend=dict(x=0, y=0, traceorder='normal', font=dict(family='sans-serif', size=12, color='white'))
                      )

    # Write the plot to an HTML file instead of returning it
    fig.write_html(output_filename, full_html=True, include_plotlyjs='cdn')
