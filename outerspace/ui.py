from ipywidgets import (
    HTML, IntSlider, FloatSlider, ToggleButtons, Dropdown, Checkbox, Button,
    VBox, ToggleButton
)
from .util import objdict

toggle_buttons_style = {'button_width': '80px'}
label_style = {'description_width': '100px'}


def create_heading(text, **kwargs):
    return HTML(value=f"<h3 style='margin-block-end: 0'>{text}</h3>", **kwargs)


def create_text(**kwargs):
    return HTML(
        style=label_style,
        **kwargs)


def create_int_slider(**kwargs):
    return IntSlider(
        style=label_style,
        continuous_update=False,
        **kwargs)


def create_toggle_buttons(**kwargs):
    return ToggleButtons(
        style=dict(**toggle_buttons_style, **label_style),
        **kwargs)


def create_float_slider(**kwargs):
    return FloatSlider(
        style=label_style,
        continuous_update=False,
        **kwargs)


def create_dropdown(**kwargs):
    return Dropdown(
        style=label_style,
        **kwargs
    )


def create_checkbox(**kwargs):
    return Checkbox(
        **kwargs,
        style=label_style
    )


def create_toggle_button(**kwargs):
    return ToggleButton(
        **kwargs,
        style=label_style
    )


def create_button(**kwargs):
    return Button(**kwargs)


def create_vbox(**kwargs):
    return VBox(**kwargs)


def create_widget(type, **kwargs):
    mapping = dict(
        heading=create_heading,
        text=create_text,
        int_slider=create_int_slider,
        float_slider=create_float_slider,
        toggle_buttons=create_toggle_buttons,
        dropdown=create_dropdown,
        checkbox=create_checkbox,
        toggle_button=create_toggle_button,
        button=create_button,
        vbox=create_vbox,
    )
    widget = mapping[type](**kwargs)
    return widget


def create_widgets(widget_params_list):
    widgets = [(params.pop('name'), create_widget(**params))
               for params in widget_params_list]
    return objdict(widgets)
