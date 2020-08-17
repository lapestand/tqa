from src.Helper import Properties
import os
import subprocess
from urllib.parse import urlparse


# def internet_on(hostname):
#     return True if os.system("ping -c 1 " + hostname) == 0 else False
#
#
# # print(internet_on("stackoverflow.com"))
# # print("https://stackoverflow.com/questions/2535055/check-if-remote-host-is-up-in-python")
# # print(urlparse("https://stackoverflow.com/questions/2535055/check-if-remote-host-is-up-in-python").netloc)
#
# address = "https://stackoverflow.com/questions/2535055/check-if-remote-host-is-up-in-python"
# on_web = True
#
#
# def make_name(address, on_web):
#     type = "pdf"
#     address = address.lower()
#     if address[-1] == '/':
#         address = address[:-1]
#     if not address.endswith(tuple(Properties.supported_file_types)):
#         address = address + '.' + type
#     address = address.split('/')[-1] if on_web else os.path.basename(address)
#     return address
#
#
# address = make_name(address, on_web)
#
# print("Address is: " + address)
# print(address.split('/')[-1] if on_web else os.path.basename(address))
#
# with open(os.devnull, 'w') as DEVNULL:
#     try:
#         subprocess.check_call(
#             ['ping', '-c', '1', 'stackoverflow.com'],
#             stdout=DEVNULL,  # suppress output
#             stderr=DEVNULL
#         )
#         is_up = True
#     except subprocess.CalledProcessError:
#         is_up = False
#
# print(is_up)


print("FILE TYPE ✖")
print("FILE TYPE ✗")
print("FILE TYPE ✔")