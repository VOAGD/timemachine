#!/usr/bin/env python
# coding: utf-8

# In[ ]:


data = input(" :")

array = bytearray(data, "utf8")
byte_list = []

for byte in array:
    binary= bin(byte)
    byte_list.append(binary)

print(byte_list)


# In[ ]:




