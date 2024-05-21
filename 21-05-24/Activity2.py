# create a parent class vehicle with some 4 child class with the concept of polymorphism

class Vehicle:
    def move(self):
        raise NotImplementedError("Subclasses must implement this method")

class Car(Vehicle):
    def move(self):
        return "The car is driving on the road."

class Bike(Vehicle):
    def move(self):
        return "The bike is pedaling on the trail."

class Truck(Vehicle):
    def move(self):
        return "The truck is hauling goods on the highway."

class Bus(Vehicle):
    def move(self):
        return "The bus is transporting passengers in the city."


vehicles = [Car(), Bike(), Truck(), Bus()]


for vehicle in vehicles:
    print(vehicle.move())
