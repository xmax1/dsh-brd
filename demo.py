"""
Create a dashboard using Panel
Dashboard is used to configure a pydantic class

One page with subsections
Each subsection contains a subclass

List of strings: Drop down menu
List with numbers: Numerical edit of numbers (may have to convert to string?)
Numbers: Numerical edit
Bool: True / False switch

All variables (apart from bool) need possibility to be None (empty)
"""

from typing import List  # in python 3.10 this would be not needed, but we are in python 3.8
from pydantic import BaseModel


class Cfg(BaseModel):

  string_list: List[str] = []
  float_array: List[List[float]] = []
  test: bool = True

  class App(BaseModel):
    name: str = "default_name1"
    value1: int = 0
    value2: float = 0.0

  class Model(BaseModel):
    title: str = "default_title"
    count: int = 0

  app: App = App()
  model: Model = Model()


data = {
  "app": {
    "name": "example",
    "value1": 42,
    "value2": 3.14
  },
  "model": {
    "title": "test",
    "count": 10
  },
  "string_list": ["hello", "world"],
  "float_array": [[0.2, 0.1], [0.9, 1.2]]
}

c = Cfg(**data)

print(c)
