from rookCalculation import RookCalaulation

class PieceFactory:
    @staticmethod
    def create_piece(piece_type):
        if piece_type == "rook":
            return RookCalaulation()
        # elif piece_type == "pawn":
        #     return Pawn(color)
        # elif piece_type == "knight":
        #     return Knight(color)
        else:
            raise ValueError(f"Unknown piece type: {piece_type}")
