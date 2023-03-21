import panel as pn
from pydantic import BaseModel, Field
from typing import List, Tuple

# Define the configuration model
class ConfigModel(BaseModel):
    param1: str = Field("default_value1", description="Description for param1")
    param2: int = Field(0, description="Description for param2")
    param3: List[str] = Field([], description="Description for param3")
    param4: Tuple[float, float] = Field((0.0, 0.0), description="Description for param4")

# Define the configuration form
config_form = pn.widgets.form.Form(
    pn.widgets.TextInput(name="param1", width=300),
    pn.widgets.IntInput(name="param2", width=300),
    pn.widgets.CheckboxGroup(name="param3", options=["option1", "option2", "option3"], width=300),
    pn.widgets.RangeSlider(name="param4", start=0.0, end=1.0, value=(0.0, 0.0), step=0.1, width=300),
    width=350
)

# Define the configuration panel
config_panel = pn.Column(
    pn.layout.HSpacer(height=50),
    pn.pane.Markdown("### Configuration Settings"),
    config_form,
    pn.layout.HSpacer(height=50),
    width=400
)

# Define the callback function
def update_config(event):
    config = ConfigModel(
        param1=config_form.param1.value,
        param2=config_form.param2.value,
        param3=list(config_form.param3),
        param4=tuple(config_form.param4.value)
    )
    # Do something with the updated config object
    print(config)

# Attach the callback to the form submit event
config_form.on_submit(update_config)

# Display the configuration panel
config_panel.show()
