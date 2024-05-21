# Create a class 'Shape' with a method 'area()'. 
# Then create two subclasses, 'Rectangle' and 'Circle',
# which override the 'area()' method to calculate the area of their respective shapes.

import math

class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement this method")

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

# Example usage
rectangle = Rectangle(width=5, height=10)
print(f"Rectangle area: {rectangle.area()}")  
circle = Circle(radius=7)
print(f"Circle area: {circle.area()}")  
