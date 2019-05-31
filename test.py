import matplotlib.pyplot as plt
import numpy as np
import pickle
import socket

HOST = "127.0.0.1"
PORT = 4321

def recv_object(client):
    packets = []
    while True:
        packet = client.recv(1024)
        if not packet: break
        packets.append(packet)
    object = pickle.loads(b"".join(packets))
    return object

with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as client:
    client.connect((HOST, PORT))
    rgbd = recv_object(client)
    rgb, d = rgbd[..., :3], rgbd[..., 3]
    plt.imshow(rgb)
    # plt.imshow(d)
    plt.show()
