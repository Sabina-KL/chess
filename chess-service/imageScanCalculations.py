
from piece_factory import PieceFactory
def scan_image(data):
    # Perform image scan logic here
    # `data` could be a list of pixel values or any other image-related data
    result = f"Scanned image with data: {data}"
    return result


def scan_pieces(file):
    # Using the factory to create a piece
    piece = PieceFactory.create_piece("rook")

    # Output the movement logic of the piece
    print(f"Created rook.")