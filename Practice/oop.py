class Car():  #starts with capital letters
   
   def __init__(self, brand, model):  #starting constructor/method, start with self for every method
      self.brand = brand
      self.model = model

      
   
my_car = Car('BMW', 'x3')  #my_car is obj and Car is class
print(my_car.brand)
print(my_car.model)


class Employee:
    company = "Google"  # Class variable → shared by all objects

    def __init__(self, name):
        self.name = name  # Instance variable → unique to each object

e1 = Employee("Alice")
e2 = Employee("Bob")

print(e1.company)
print(e2.company)

#Encapsulation Hide internal variables. Allow controlled access.

class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # private variable

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance
    
#How to access   
acc = BankAccount(1000)

acc.deposit(500)
print(acc.get_balance())   # ✅ allowed

#print(acc.__balance)   # ❌ Error

#Abstraction: we can not create the Obj of abst class

from abc import ABC, abstractmethod

class Shape(ABC):

    @abstractmethod
    def area(self):
        pass
    

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius
    
c = Circle(5)
print(c.area())  # ✅ works

#s = Shape()  # ❌ Error


#Inheritance = Child class gets properties of parent.

class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def bark(self):
        print("Dog barks")

d = Dog()
d.speak()   # ✅ inherited
d.bark()    # ✅ own method

#Note: If parent has private variable, Child cannot directly access

class Animal:
    def __init__(self):
        self.__secret = "hidden"

# class Dog(Animal):
#     def show(self):
#         print(self.__secret)  # ❌ error

# use Super() method

class Dog(Animal):
    def __init__(self):
        super().__init__()




#Polymorphism = One interface, multiple forms.

class Animal:
    def sound(self):
        print("Generic sound")

class Dog(Animal):
    def sound(self):
        print("Bark")


a = Animal()
d = Dog()

a.sound()   # Generic sound
d.sound()   # Bark
