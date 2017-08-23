import hid as hid
import time

hidraw = hid.device(0x1234,0xed02)
hidraw.open(0x1234, 0xed02)

print ("Serial:"+str(hidraw.get_serial_number_string()))

serial_number = 'SN20120229000290'

print ("Reading data...")

hidraw.set_nonblocking(True)
counter=0;
while(True):

    counter=counter+1
    data = hidraw.read(34,100)
    print ("Reading "+str(len(data))+" bytes from emotiv...")
    time.sleep(0.1)
    if (counter%100==0):
        print ("Reloading")
        hidraw.close()
        time.sleep(2)
        hidraw = hid.device(0x1234,0xed02)
        hidraw.open(0x1234, 0xed02)
        time.sleep(2)
        hidraw.set_nonblocking(True)
    if (counter>1000 or len(data)>0):
        break

hidraw.close()
