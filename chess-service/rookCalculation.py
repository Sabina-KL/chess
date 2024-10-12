# rook.py
from pieceCalculation import PieceCalculation  # Import Piece class from piece.py

# Concrete class for Rook
class RookCalaulation(PieceCalculation):
    def calculate(self):
        return "Rook moves horizontally or vertically."