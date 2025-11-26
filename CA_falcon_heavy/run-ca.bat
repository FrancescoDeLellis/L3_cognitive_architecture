@echo off
cd /d E:\Repositories\SHARESPACE\L3_cognitive_architecture\CA_falcon_heavy


start "CA1" cmd /k python3 L3_server_dual_socket.py 60000 60001
start "CA2" cmd /k python3 L3_server_dual_socket.py 60010 60011
start "CA3" cmd /k python3 L3_server_dual_socket.py 60020 60021
