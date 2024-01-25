import os
import socket
import base64
import base58
import hashlib
import uuid
import json
import time

from cryptography.hazmat.primitives import serialization, hashes, padding
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

keyfile_user = '.token_user.pem'
prikey_user = ''
DID = ''
DID_sys = ''
DID_claims = {}

def init_local_did(name):
    global keyfile_user, prikey_user, DID, DID_sys, DID_claims

    keyfile_user_path = os.path.abspath(f'./{keyfile_user}')
    try:
        if os.path.exists(keyfile_user_path):
            with open(keyfile_user_path, "rb") as key_file:
                prikey_user = serialization.load_pem_private_key(key_file.read(), f'{os.path.abspath(__file__)}@{socket.gethostname()}'.encode("utf-8"))
        else:
            prikey_user = ed25519.Ed25519PrivateKey.generate()
            private_bytes = prikey_user.private_bytes( 
                encoding=serialization.Encoding.PEM,   
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(f'{os.path.abspath(__file__)}@{socket.gethostname()}'.encode("utf-8"))
            )
            with open(keyfile_user_path, "wb") as key_file:
                key_file.write(private_bytes)
        if not isinstance(prikey_user, ed25519.Ed25519PrivateKey):
            raise TypeError
    except Exception as e:
        print(f'Load user private key file [{keyfile_user_path}] failed')
        print(e)

    mac_address = ':'.join(hex(uuid.getnode())[2:].zfill(12)[i:i + 2] for i in range(0, 12, 2))
    mac_address_sha256 = base64.b64encode(hashlib.new('sha256', f'{name}:mac_address:{mac_address}'.encode('utf-8')).digest()).decode('utf-8')
    telephone_sha256 = base64.b64encode(hashlib.new('sha256', f'{name}:telephone:-'.encode('utf-8')).digest()).decode('utf-8')
    face_image_sha256 = base64.b64encode(hashlib.new('sha256', f'{name}:face_image:-'.encode('utf-8')).digest()).decode('utf-8')
    file_hash_sha256 = base64.b64encode(hashlib.new('sha256', f'{name}:file_hash:-'.encode('utf-8')).digest()).decode('utf-8')
    fingerprint = {"mac_address": mac_address_sha256, "telephone": telephone_sha256, "face_image": face_image_sha256, "file_hash": file_hash_sha256}
    fingerprint_sha256 = base64.b64encode(hashlib.new('sha256', f'{name}:fingerprint:{format_claim(fingerprint)}'.encode('utf-8')).digest()).decode('utf-8')
    exchange_key = base64.b64encode(x25519.X25519PrivateKey.from_private_bytes(
        prikey_user.private_bytes(  encoding=serialization.Encoding.Raw,
                                    format=serialization.PrivateFormat.Raw,
                                    encryption_algorithm=serialization.NoEncryption())
    ).public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)).decode('utf-8')
    verify_key = base64.b64encode(prikey_user.public_key().public_bytes(encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)).decode('utf-8')
    did_claim={"name": name, "verify_key": verify_key, "exchange_key": exchange_key, "fingerprint": {"hash": fingerprint_sha256, "info": fingerprint}}
    claim_sha256 = hashlib.new('sha256', format_claim(did_claim).encode('utf-8')).digest()
    DID = base58.b58encode_check(hashlib.new('ripemd160', claim_sha256).digest()).decode('utf-8')
    print(f'[TokenDID] Generated local_did: "{DID}"')
    DID_claims[DID] = did_claim

    with open(os.path.abspath(f'./.user_{DID}.did'), "w", encoding="utf-8") as did_file:
        json.dump(did_claim, did_file)

    try:
        folder_path = os.path.dirname(__file__)
        exensions = '.did'
        name_filter = '.token_'
        if not os.path.isdir(folder_path):
            raise ValueError("Folder path is not a valid directory.")
        filenames = []
        for filename in os.listdir(folder_path):
            _, file_extension = os.path.splitext(filename)
            if (exensions == None or file_extension.lower() in exensions) and (name_filter == None or name_filter in _):
                filenames.append(filename)
        didfiles = sorted(filenames, key=lambda x: -1 if os.sep in x else 1)
        count = len(didfiles)
        for d in didfiles:
            if (d.startswith(name_filter)):
                if (d.startswith(name_filter+'sys_')):
                    did = d.split('_')[2][:-4]
                    DID_sys = did
                else:
                    did = d.split('_')[1][:-4]
                try:
                    base58.b58decode_check(did.encode('utf-8'))
                except ValueError as e:
                    print(f'[TokenDID] Invalid checksum: {d} - {did}')
                    count -= 1
                with open(os.path.join(folder_path, d), "r", encoding="utf-8") as json_file:
                    DID_claims[did] = json.load(json_file)
        print(f'[TokenDID] Loaded {count} DIDs from root_path.')
    except Exception as e:
        print(f'Load system DIDs file failed')
        print(e)
    return


def urlsafe_patch(text, method):
    if method == 'encode':
        return text.trip(r'=+')
    elif method == 'decode':
        return text + "===="[:len(text)%4]
    else:
        return text


def encrypt(did, text):
    global prikey_user, DID_claims
    system_name = b'model_file_hub_sys'
    if did not in DID_claims:
        return None
    exkey = x25519.X25519PublicKey.from_public_bytes(base64.b64decode(DID_claims[did]["exchange_key"].encode('utf-8')))
    prikey_self = x25519.X25519PrivateKey.from_private_bytes( 
        prikey_user.private_bytes(  encoding=serialization.Encoding.Raw, \
                                    format=serialization.PrivateFormat.Raw, \
                                    encryption_algorithm=serialization.NoEncryption())
    )
    salt0 = hashlib.new('sha256',f'now()={int(time.time()/600)}'.encode('utf-8')).digest()
    derivedkey = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt0[:16],    
        info=system_name,
    ).derive(prikey_self.exchange(exkey))
    iv = os.urandom(16)
    encryptor = Cipher(algorithms.AES(derivedkey), modes.CBC(iv)).encryptor()
    padder = padding.PKCS7(128).padder()
    ct = iv + encryptor.update(padder.update(text.encode('utf-8')) + padder.finalize()) + encryptor.finalize()
    return urlsafe_patch(base64.urlsafe_b64encode(ct).decode('utf-8'), "encode")


def encrypt_default(text):
    global DID_sys
    if DID_sys:
        return encrypt(DID_sys, text)
    return None


def decrypt(did, text):
    global prikey_user, DID_claims
    system_name = b'model_file_hub_sys'
    if did not in DID_claims:
        return None
    exkey = x25519.X25519PublicKey.from_public_bytes(base64.b64decode(DID_claims[did]["exchange_key"].encode('utf-8')))
    prikey_self = x25519.X25519PrivateKey.from_private_bytes(
        prikey_user.private_bytes(  encoding=serialization.Encoding.Raw, \
                                    format=serialization.PrivateFormat.Raw, \
                                    encryption_algorithm=serialization.NoEncryption())
    )
    time_value = int(time.time()/600)
    salt0 = hashlib.new('sha256',f'now()={time_value}'.encode('utf-8')).digest()
    derivedkey = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt0[:16],
        info=system_name,
    ).derive(prikey_self.exchange(exkey))
    ct = base64.urlsafe_b64decode(urlsafe_patch(text, "decode").encode('utf-8'))
    decryptor = Cipher(algorithms.AES(derivedkey), modes.CBC(ct[:16])).decryptor()
    unpadder = padding.PKCS7(128).unpadder()
    flag = False
    try:
        pt = unpadder.update(decryptor.update(ct[16:]) + decryptor.finalize()) + unpadder.finalize()
        pt = pt.decode('utf-8')
    except (ValueError, UnicodeDecodeError) as e:
        flag = True
    if flag:
        salt0 = hashlib.new('sha256',f'now()={time_value+1}'.encode('utf-8')).digest()
        derivedkey = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt0[:16],
            info=system_name,
        ).derive(prikey_self.exchange(exkey))
        ct = base64.urlsafe_b64decode(urlsafe_patch(text, "decode").encode('utf-8'))
        decryptor = Cipher(algorithms.AES(derivedkey), modes.CBC(ct[:16])).decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        try:
            pt = unpadder.update(decryptor.update(ct[16:]) + decryptor.finalize()) + unpadder.finalize()
            pt = pt.decode('utf-8')
        except (ValueError, UnicodeDecodeError) as e:
            pt = None
    return pt


def decrypt_default(text):
    global DID_sys
    if DID_sys:
        return decrypt(DID_sys, text)
    return None


def verify_did_claim(o_did, o_claim):
    claim_sha256 = hashlib.new('sha256', format_claim(o_claim).encode('utf-8')).digest()
    did = base58.b58encode_check(hashlib.new('ripemd160', claim_sha256).digest()).decode('utf-8')
    return o_did == did


def sign(data):
    global prikey_user
    return base64.b64encode(prikey_user.sign(data.encode('utf-8'))).decode('utf-8')


def verify(signature, text):
    global prikey_user
    return prikey_user.public_key().verify(base64.b64decode(signature.encode('utf-8')), text.encode('utf-8'))


def verify_by_DID(signature, text, did):
    return ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(DID_claims[did]["verify_key"].encode('utf-8'))).verify(base64.b64decode(signature.encode('utf-8')), text.encode('utf-8'))
    

def format_claim(claim):
    keys = sorted(list(claim.keys()))
    text = ''
    for k in keys:
        info = claim[k]
        text += f'{k}:{info},'
    return text[:-1]


def get_register_claim(sys_name):
    global DID, DID_claims
    claim = {
        'sys_name': f'{sys_name}',
        'user_claim': f'Register for {sys_name} with my DID claim',
        'user_did': DID,
        'user_did_claim': json.dumps(DID_claims[DID])
    }
    claim.update({'user_sign': sign(format_claim(claim))})
    return json.dumps(claim)


