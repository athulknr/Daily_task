#print maximum & minimum speed
class Vehicle:
    
    color = "White"
   
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage

car1 = Vehicle(120, 15000)
car2 = Vehicle(150, 20000)

print(f"Car 1: Max Speed: {car1.max_speed} km/h, Mileage: {car1.mileage} km, Color: {car1.color}")
print(f"Car 2: Max Speed: {car2.max_speed} km/h, Mileage: {car2.mileage} km, Color: {car2.color}")

print(f"Car 1 color is the same as Car 2 color: {car1.color == car2.color}")

Vehicle.color = "Red"
print(f"Car 1: Max Speed: {car1.max_speed} km/h, Mileage: {car1.mileage} km, Color: {car1.color}")
print(f"Car 2: Max Speed: {car2.max_speed} km/h, Mileage: {car2.mileage} km, Color: {car2.color}")
