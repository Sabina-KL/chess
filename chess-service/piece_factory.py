from pieces import Rook, Pawn, Knight

class PieceFactory:
    @staticmethod
    def create_piece(piece_type, color):
        if piece_type == "rook":
            return Rook(color)
        # elif piece_type == "pawn":
        #     return Pawn(color)
        # elif piece_type == "knight":
        #     return Knight(color)
        else:
            raise ValueError(f"Unknown piece type: {piece_type}")
