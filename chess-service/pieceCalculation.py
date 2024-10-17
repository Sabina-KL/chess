from abc import ABC, abstractmethod

# Abstract class for chess pieces
class PieceCalculation(ABC):
    def __init__(self, color="white"):
        self.color = color  # Each piece will have a color (e.g., 'white' or 'black')
        self._file = None  # Initialize file with None or any default value

    # Getter for file
    @property
    def file(self):
        """Getter for file"""
        return self._file

    # Setter for file
    @file.setter
    def file(self, new_file):
        """Setter for file"""
        if isinstance(new_file, str):  # Validation for file being a string
            self._file = new_file
        else:
            raise ValueError("file should be a string")

    @abstractmethod
    def calculate(self):
        pass
