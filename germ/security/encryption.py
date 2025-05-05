from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import hashlib
import os


def encrypt_integer(n: int, key: bytes) -> str:
    # Convert int to 8-byte big-endian binary
    data = n.to_bytes(8, byteorder='big')

    # Apply PKCS7 padding to make it 16 bytes (AES block size)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    # Generate a random IV
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()

    # Return IV + ciphertext as hex
    return (iv + ciphertext).hex()


def decrypt_integer(hex_str: str, key: bytes) -> int:
    data = bytes.fromhex(hex_str)
    iv = data[:16]
    ciphertext = data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove PKCS7 padding
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()

    # Convert back to integer
    return int.from_bytes(data, byteorder='big')


def derive_key_from_passphrase(passphrase: str) -> bytes:
    return hashlib.sha256(passphrase.encode('utf-8')).digest()  # 32-byte AES-256 key
