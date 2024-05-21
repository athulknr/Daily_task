class Vehicle:
    def __init__(self, color, max_speed):
        self.color = color
        self.max_speed = max_speed

    def display(self):
        print("Color:", self.color)
        print("Max Speed:", self.max_speed)


class Bus(Vehicle):
    def __init__(self, color, max_speed, seating_capacity=50):
        super().__init__(color, max_speed)
        self.seating_capacity = seating_capacity

    def seating_capacity_info(self):
        print("Seating Capacity:", self.seating_capacity)



my_bus = Bus("Red", 80)

my_bus.display() 
my_bus.seating_capacity_info() 
