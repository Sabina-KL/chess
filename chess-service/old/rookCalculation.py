#import an abstract class for inheritance
from pieceCalculation import PieceCalculation
# Concrete class for Rook
class RookCalaulation(PieceCalculation):
    def calculate(self):
        return "Rook moves horizontally or vertically."