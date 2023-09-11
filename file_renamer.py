import os

path = 'data/D/'

i = 0
for filename in os.listdir(path):
    new_name = "Image{}.jpg".format(i)
    my_source = path + filename
    my_dest = path + new_name
    os.rename(my_source, my_dest)
    i += 1