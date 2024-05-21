class Vehicle:
    def __init__(self, max_speed, mileage):
        self.max_speed = max_speed
        self.mileage = mileage


car = Vehicle(120, 15000)
print(f"Max Speed: {car.max_speed} km/h")
print(f"Mileage: {car.mileage} km")



class Vehicle:
    pass


vehicle = Vehicle()
print(vehicle)
