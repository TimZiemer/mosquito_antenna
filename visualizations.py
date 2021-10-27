import plotly.graph_objects as go
from plotly.subplots import make_subplots


def animate_antenna_movement(antenna_coords, capsule_coords) -> None:
    fig = go.Figure(
        data=[go.Scatter(x=[cp[0] for cp in capsule_coords], y=[cp[1] for cp in capsule_coords],
                         name="Antenna", line=dict(width=4, color="orange")),
              go.Scatter(x=[cp[0] for cp in capsule_coords], y=[cp[1] for cp in capsule_coords],
                         name="Capsule", line=dict(width=4, color="blue"))],
        layout=go.Layout(
            xaxis=dict(range=[-1.5, 1.5], autorange=False),
            yaxis=dict(range=[-1.5, 1.5], autorange=False),
            title="Animated antenna movement for given signal",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Move antenna",
                              method="animate",
                              args=[None])])]
        ),
        frames=[go.Frame(
            data=[go.Scatter(
                x=[cp[0] for cp in arr],
                y=[cp[1] for cp in arr],
            )]
        ) for arr in antenna_coords]
    )
    fig.show()


def create_slider_for_time_series_and_db_spec(time_series, input_signal, db_specs, db_specs_orig, x_title = None) -> None:

    fig = make_subplots(rows=1, cols=2,
                        x_title=x_title,
                        specs=[[{"secondary_y": True},{}]])

    # Create graphs for each prong
    for i, series in enumerate(time_series):  # Time Series
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#003153", width=4),
                name=f"Prong {i}",
                y=series), row=1, col=1)
    for i, series in enumerate(time_series):  # Antenna spec
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#003153", width=4),
                name = f"Antenna frequencies, Prong {i}",
                x=db_specs[i][0],
                y=db_specs[i][1]), row=1, col=2)
    for _ in range(time_series.shape[0]):  # Orig spec
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="orange", width=4),
                name = f"Input frequencies",
                opacity=0.5,
                x=db_specs_orig[0][0],
                y=db_specs_orig[0][1]), row=1, col=2)
    for _ in range(time_series.shape[0]):  # Orig Wave
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="orange", width=4),
                name = f"Orig signal",
                opacity=0.5,
                y=input_signal), row=1, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Samples", row=1, col=1)
    fig.update_yaxes(title_text="Elongation", row=1, col=1)
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=2, )
    fig.update_yaxes(title_text="Amplitude [dB]", row=1, col=2, range=[-50, max(db_specs_orig[0][1])])

    # Make 0th prong visible
    fig.data[0].visible = True
    fig.data[time_series.shape[0]].visible = True
    fig.data[time_series.shape[0]*2].visible = True
    fig.data[time_series.shape[0]*3].visible = True

    steps = []
    num_steps = len(time_series)
    for i in range(num_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Antenna movement prong: " + str(i)}],
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i+num_steps] = True
        step["args"][0]["visible"][i+(num_steps*2)] = True
        step["args"][0]["visible"][i+(num_steps*3)] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Prong: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        legend=dict(
            yanchor="top",
            x=0.94,
            xanchor="right",
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        font=dict(
            family="serif",
            size=17,
        )
    )

    # This can be used to take one trace out of the legend (mostly for the purpose of creating plots for papers)
    for trace in fig['data']:
        if ("Never_gonna_give_you_up..." not in trace['name']): trace['showlegend'] = False

    fig.show()


def create_multicolor_plot_for_paper(m1_spec, m2_spec, input_spec, output_spec) -> None:
    """
    Creates a plot in which the separate input sounds are colored differently
    :param m1_spec: Mosquito-Sound 1
    :param m2_spec: Mosquito-Sound 2
    :param input_spec: Overlayed mosquito sounds
    :param output_spec: Nonlinear antenna output
    :return: None
    """
    specs_to_display = [m1_spec, m2_spec]
    colors = ["#ACDF87", "#FFC6C4"]
    names = ["Aegypti Male", "Aegypti Female"]

    fig = make_subplots(rows=1, cols=1,
                        x_title=None)

    for i in range(40):
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#946587", width=4),
                name=f"Prong {i + 1}",
                x=output_spec[i][0],
                y=output_spec[i][1]), row=1, col=1)

    for spec, c, name in zip(specs_to_display, colors, names):
        for i in range(40):
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color=c, width=4),
                    name = name,
                    x=spec[0][0],
                    y=spec[0][1]), row=1, col=1)

    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1, range=[0, 2200])
    fig.update_yaxes(title_text="Amplitude [dB]", row=1, col=1, range=[-20, max(input_spec[0][1])])

    fig.data[0].visible = True
    fig.data[40].visible = True
    fig.data[40*2].visible = True

    steps = []
    num_steps = 40
    for i in range(num_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Antenna movement prong: " + str(i)}],
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i+num_steps] = True
        step["args"][0]["visible"][i+(num_steps*2)] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Prong: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        legend=dict(
            yanchor="top",
            x=1,
            xanchor="right",
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        font=dict(
            family="serif",
            size=17,
        )
    )

    fig.show()


### Following are specific antenna paper plot functions ###

def antiphase_prong_plot(p0, p25):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            line=dict(color="#003153", width=4),
            name=f"Prong 0",
            y=p0))
    fig.add_trace(
        go.Scatter(
            line=dict(color="#003153", width=4, dash="dash"),
            name=f"Prong 26",
            y=p25))
    fig.update_xaxes(title_text="Samples", range=[0,100])
    fig.update_yaxes(title_text="Elongation")
    fig.update_layout(
        legend=dict(
            yanchor="top",
            x=1,
            xanchor="right",
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        font=dict(
            family="serif",
            size=17,
        )
    )
    fig.show()


def magnitude_of_nonlinearities_plot(weaker_signal, stronger_signal):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            line=dict(color="#003153", width=4),
            name=f"Prong 0, regular",
            x=weaker_signal[0][0],
            y=weaker_signal[0][1]))
    fig.add_trace(
        go.Scatter(
            line=dict(color="#003153", width=4, dash="dash"),
            name=f"Prong 0, amplified",
            x=stronger_signal[0][0],
            y=stronger_signal[0][1]))
    fig.update_xaxes(title_text="Frequency [Hz]")
    fig.update_yaxes(title_text="Amplitude [dB]")
    fig.update_layout(
        legend=dict(
            yanchor="top",
            x=1,
            xanchor="right",
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        ),
        font=dict(
            family="serif",
            size=17,
        )
    )
    fig.show()
