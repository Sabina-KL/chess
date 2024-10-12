from abc import ABC, abstractmethod

# Abstract class for chess pieces
class PieceCalculation(ABC):
    def __init__(self, color):
        self.color = color  # Each piece will have a color (e.g., 'white' or 'black')

    @abstractmethod
    def calculate(self):
        pass