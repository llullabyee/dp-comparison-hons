        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=true_vals, mode="lines", name="Raw mains"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=gai_noise, mode="lines", name="Gai noise"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=dplm_noise, mode="lines", name="DPLM noise"
            )
        )
        fig.show()
        break
