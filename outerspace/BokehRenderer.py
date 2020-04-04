from ipywidgets import Output
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import ColumnDataSource, Label
from bokeh.plotting import figure, output_file, save


class BokehRenderer:
    def __init__(self, X, y, additional_columns, tooltips, color):
        source = ColumnDataSource(data=dict(
            x=X[:, 0],  # arbitrary
            y=X[:, 1],  # arbitrary
            label=[str(e) for e in y],
            **additional_columns))

        # create plot
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'save', 'reset']
        if tooltips:
            tools.append('hover')
        self.p = figure(
            output_backend="webgl",
            plot_height=300,
            plot_width=300,
            sizing_mode='scale_width',
            tools=tools,
            active_drag='pan',
            active_scroll='wheel_zoom',
            toolbar_location="below",
            tooltips=tooltips
          )
        self.p.toolbar.logo = None  # remove bokeh logo

        self.scatter = self.p.scatter(
            source=source,
            x='x',
            y='y',
            color=color,
            legend=False,
            line_width=0,
            alpha=0.8,
            size=5)
        self.scatter.visible = False

        # status message on top of graph
        self.status_message = Label(x=20, y=20, x_units='screen',
                                    y_units='screen', text='\N{HOURGLASS}',
                                    render_mode='css', text_font_size='40px')
        self.status_message.visible = False
        self.p.add_layout(self.status_message)
        
    def pre_update(self):
        push_notebook(self.handle)
        
    def show_status(self, msg):
        self.status_message.text = f'\N{HOURGLASS} {msg}'
        self.status_message.visible = True
        push_notebook(self.handle)
        
    def hide_status(self):
        self.status_message.visible = False
        push_notebook(self.handle)
        
    def show_plot(self, x, y):
        self.scatter.visible = True
        self.scatter.data_source.data['x'] = x
        self.scatter.data_source.data['y'] = y
        push_notebook(self.handle)
        
    def clear_plot(self):
        self.scatter.visible = False
        push_notebook(self.handle)
        
    def save_plot(self, path):
        output_file(path)
        save(self.p)
        
    def _ipython_display_(self, **kwargs):
        output_notebook(hide_banner=True)
        self.handle = show(self.p, notebook_handle=True)