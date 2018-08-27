#coding: latin-1

# This program works in tandem with "ShowSensorData"
# This program will log all the information that can be seen with the other program

# https://github.com/walac/pyusb/blob/master/docs/tutorial.rst

import hid
import time

for d in hid.enumerate(0, 0):
    keys = d.keys()
    keys.sort()
    for key in keys:
        print "%s : %s" % (key, d[key])
        print ""

# For Mac OS, try to unload the AppleUSBFtdi kext:
#
# sudo kextunload -bundle-id com.apple.driver.AppleUSBFTDI
#
# You can reload it with:
#
# sudo kextload -bundle-id com.apple.driver.AppleUSBFTDI
#
# Source:
# http://pylibftdi.readthedocs.io/en/0.16.0/troubleshooting.html
