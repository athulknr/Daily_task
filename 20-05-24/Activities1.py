class Adit:
    def __init__(self):
        self.name = None
        self.phone_number = None

    def store_info(self, name, phone_number):
        self.name = name
        self.phone_number = phone_number

    def print_info(self):
        if self.name is not None and self.phone_number is not None:
            print("Name:", self.name)
            print("Phone Number:", self.phone_number)
        else:
            print("Name and phone number not provided.")


# Example usage:
adit_instance = Adit()
adit_instance.store_info("Adit", "1234567890")
adit_instance.print_info()